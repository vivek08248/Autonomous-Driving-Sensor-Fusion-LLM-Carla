"""
ad_llm.py - LLM reasoning module + DAgger arbitration (DriveGPT4-V2, Sec. 3.1-3.2).

  * build_prompt()   compact, structured scene description (~150-250 tokens).  The old
                     prompt was ~960 tokens and Ollama truncated it to the LAST 194 tokens,
                     so the model never saw most of the scene.
  * LLMWorker        one background thread, keep-alive HTTP connection to Ollama,
                     never blocks the 20 Hz control loop, always uses the NEWEST scene.
  * DaggerArbiter    paper's on-policy supervision: the privileged expert compares its
                     decision with the LLM's; above the discrepancy threshold the expert
                     takes over and the sample is aggregated into dagger_dataset.jsonl
                     (expert output = label) for later fine-tuning.
"""
import http.client
import json
import os
import re
import threading
import time
import urllib.parse

try:
    import config as _cfg
except Exception:
    _cfg = None


def _c(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


ACTIONS = ("KEEP", "FOLLOW", "SLOW", "STOP", "YIELD", "BYPASS_LEFT", "BYPASS_RIGHT")

SYSTEM_PROMPT = (
    "You are the brain of an autonomous vehicle (DriveGPT4-V2 planner). "
    "Input: ego state, target point and fused sensor objects in the road frame "
    "(s = metres ahead of the front bumper, d = metres right of our lane centre, v = km/h). "
    "Rules: never hit anything; stop before the line at red or stoppable yellow lights; "
    "yield to pedestrians, cyclists and animals in or entering our lane; keep 2 s headway "
    "behind moving vehicles and 4 m when stopped; pass a stopped obstacle only if the passing "
    "lane is free, preferring our own same-direction lanes (opposite lane only for road works with no "
    "oncoming traffic); slow down in rain, fog, night and junctions; otherwise drive at the speed limit. "
    'Reply ONLY with JSON: {"action": "KEEP|FOLLOW|SLOW|STOP|YIELD|BYPASS_LEFT|BYPASS_RIGHT", '
    '"target_speed": <km/h>, "reason": "<max 5 words>"}'
)

INTENT_TO_ACTION = {
    "KEEP_LANE": "KEEP", "FOLLOW": "FOLLOW", "SLOW_DOWN": "SLOW", "STOP": "STOP",
    "EMERGENCY_STOP": "STOP", "YIELD": "YIELD", "BYPASS_LEFT": "BYPASS_LEFT",
    "BYPASS_RIGHT": "BYPASS_RIGHT", "LANE_CHANGE_LEFT": "BYPASS_LEFT",
    "LANE_CHANGE_RIGHT": "BYPASS_RIGHT", "EVADE_LEFT": "BYPASS_LEFT",
    "EVADE_RIGHT": "BYPASS_RIGHT", "RETURN_LANE": "KEEP", "REVERSE": "STOP",
}


def build_prompt(t, weather, ego_kmh, speed_limit_kmh, route_shape, junction_m, target_point,
                 light, objs, ego_front_s, ego_d, max_objs=6, lane_pos=None):
    """Compact scene text.  objs = list of ad_types.Obj (route frame)."""
    lines = [f"weather={weather} ego_speed={ego_kmh:.0f}km/h limit={speed_limit_kmh:.0f}km/h"]
    if lane_pos:
        lines.append(f"ego lane {lane_pos[0]} of {lane_pos[1]} same-direction lanes (1 = next to oncoming)")
    lines.append(f"route: {route_shape}" + (f", junction in {junction_m:.0f}m" if junction_m is not None else ""))
    lines.append(f"target_point: s={target_point[0]:.0f} d={target_point[1]:+.1f}")
    if light is not None:
        lines.append(f"traffic_light: {light.state} stop line {light.s_line - ego_front_s:.0f}m")
    else:
        lines.append("traffic_light: none")
    rel = []
    for o in objs:
        s_rel = o.s - ego_front_s
        if s_rel < -8.0 or s_rel > 70.0:
            continue
        if abs(o.d - ego_d) > 9.0:
            continue
        rel.append((abs(s_rel) + 2.0 * abs(o.d - ego_d), o, s_rel))
    rel.sort(key=lambda r: r[0])
    if not rel:
        lines.append("objects: none")
    else:
        lines.append("objects:")
        for _, o, s_rel in rel[:max_objs]:
            in_lane = abs(o.d - ego_d) < o.half_wid + 1.3
            motion = "static" if o.speed < 0.3 else (
                "oncoming" if o.vs < -0.5 else ("crossing" if abs(o.vd) > 0.4 else "moving"))
            name = {"vehicle": "car", "walker": "pedestrian"}.get(o.kind, o.kind)
            lines.append(f"- {name} s={s_rel:.0f} d={o.d:+.1f} v={o.vs * 3.6:.0f} {motion}"
                         f"{' IN_LANE' if in_lane else ''}")
    return "\n".join(lines)


def parse_reply(text):
    if not text:
        return None
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    act = str(obj.get("action", obj.get("intent", "KEEP"))).upper().strip()
    act = act.replace("KEEP_LANE", "KEEP").replace("SLOW_DOWN", "SLOW")
    if act not in ACTIONS:
        if "LEFT" in act:
            act = "BYPASS_LEFT"
        elif "RIGHT" in act:
            act = "BYPASS_RIGHT"
        elif "STOP" in act or "BRAKE" in act:
            act = "STOP"
        elif "YIELD" in act:
            act = "YIELD"
        elif "SLOW" in act:
            act = "SLOW"
        elif "FOLLOW" in act:
            act = "FOLLOW"
        else:
            act = "KEEP"
    try:
        spd = float(obj.get("target_speed", obj.get("target_speed_kmh", 0.0)))
    except Exception:
        spd = 0.0 if act in ("STOP", "YIELD") else 20.0
    if act in ("STOP",):
        spd = 0.0
    spd = max(0.0, min(60.0, spd))
    return {"action": act, "target_speed_kmh": spd, "reason": str(obj.get("reason", ""))[:60]}


class LLMWorker:
    """Single background thread; submit() never blocks; newest prompt wins."""

    def __init__(self, model, host=None):
        self.host = host or _c("OLLAMA_HOST", "http://localhost:11434")
        self.model = model
        self.timeout = float(_c("AD_LLM_TIMEOUT_S", 6.0))
        self._cv = threading.Condition()
        self._pending = None
        self._result = None
        self._busy = False
        self._stop = False
        self._conn = None
        self.last_error = ""
        self.n_ok = self.n_fail = 0
        self.last_timing = {}
        self.gpu_info = None
        self.latencies = []
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    # -------------------------------------------------------------- public
    def set_model(self, model):
        with self._cv:
            self.model = model
            self._pending = None
            self._result = None

    def warmup(self):
        self.submit("ok", tag="warmup")

    def submit(self, prompt, tag=None, meta=None):
        """meta = expert decision for the SAME scene (used as the DAgger label)."""
        with self._cv:
            self._pending = (prompt, tag, time.monotonic(), meta)
            self._cv.notify()

    @property
    def busy(self):
        return self._busy or self._pending is not None

    def poll(self):
        """Return newest finished result (dict) once, else None."""
        with self._cv:
            r, self._result = self._result, None
            return r

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    # ------------------------------------------------------------- worker
    def _connect(self):
        u = urllib.parse.urlparse(self.host)
        self._conn = http.client.HTTPConnection(u.hostname or "localhost", u.port or 11434,
                                                timeout=self.timeout)

    def _request(self, prompt, warm=False):
        if self._conn is None:
            self._connect()
        body = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": _c("LLM_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": 0.0,
                "num_predict": 4 if warm else int(_c("AD_LLM_NUM_PREDICT", 40)),
                "num_ctx": int(_c("AD_LLM_NUM_CTX", 1024)),
            },
        }
        if _c("LLM_DISABLE_THINK", True):
            body["think"] = False
        data = json.dumps(body)
        try:
            self._conn.request("POST", "/api/generate", body=data,
                               headers={"Content-Type": "application/json"})
            resp = self._conn.getresponse()
            raw = resp.read()
        except Exception:
            # reconnect once (keep-alive socket may have been closed)
            self._connect()
            self._conn.request("POST", "/api/generate", body=data,
                               headers={"Content-Type": "application/json"})
            resp = self._conn.getresponse()
            raw = resp.read()
        if resp.status != 200:
            if resp.status == 400 and b"think" in raw and "think" in body:
                body.pop("think")
                self._conn.request("POST", "/api/generate", body=json.dumps(body),
                                   headers={"Content-Type": "application/json"})
                resp = self._conn.getresponse()
                raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:120]!r}")
        js = json.loads(raw.decode("utf-8", "ignore"))
        return js.get("response", ""), js

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                prompt, tag, t_sub, meta = self._pending
                self._pending = None
                self._busy = True
                model = self.model
            t0 = time.monotonic()
            try:
                text, js = self._request(prompt, warm=(tag == "warmup"))
                lat = (time.monotonic() - t0) * 1000.0
                dec = parse_reply(text) if tag != "warmup" else None
                ns = 1e-6
                timing = {"total_ms": js.get("total_duration", 0) * ns,
                          "load_ms": js.get("load_duration", 0) * ns,
                          "prompt_ms": js.get("prompt_eval_duration", 0) * ns,
                          "gen_ms": js.get("eval_duration", 0) * ns,
                          "prompt_tok": js.get("prompt_eval_count", 0),
                          "gen_tok": js.get("eval_count", 0)}
                timing["http_ms"] = max(0.0, lat - timing["total_ms"])
                self.last_timing = timing
                if tag == "warmup":
                    self.gpu_info = ollama_ps(self.host, model)
                if tag != "warmup":
                    if dec is None:
                        self.n_fail += 1
                        self.last_error = "unparsable reply"
                    else:
                        self.n_ok += 1
                        self.latencies.append(lat)
                        if len(self.latencies) > 500:
                            self.latencies = self.latencies[-500:]
                        dec.update(latency_ms=lat, t_done=time.monotonic(), t_submit=t_sub,
                                   model=model, prompt=prompt, tag=tag, meta=meta,
                                   eval_tokens=js.get("eval_count", 0),
                                   prompt_tokens=js.get("prompt_eval_count", 0), timing=timing)
                        if self.n_ok % 20 == 1:
                            print(f"\n[LLM] {model}: {lat:.0f} ms = model load {timing['load_ms']:.0f} + "
                                  f"read prompt {timing['prompt_tok']} tok {timing['prompt_ms']:.0f} + "
                                  f"generate {timing['gen_tok']} tok {timing['gen_ms']:.0f} + "
                                  f"http {timing['http_ms']:.0f}  "
                                  f"({timing['gen_tok'] / max(1e-3, timing['gen_ms'] / 1000):.0f} tok/s)")
                        with self._cv:
                            self._result = dec
            except Exception as e:
                self.n_fail += 1
                self.last_error = str(e)[:120]
                self._conn = None
                time.sleep(0.5)
            finally:
                self._busy = False


def ollama_ps(host, model, timeout=2.0):
    """How much of the loaded model sits in GPU memory (Ollama /api/ps)."""
    try:
        u = urllib.parse.urlparse(host)
        conn = http.client.HTTPConnection(u.hostname or "localhost", u.port or 11434, timeout=timeout)
        conn.request("GET", "/api/ps")
        js = json.loads(conn.getresponse().read().decode("utf-8"))
        for m in js.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size, vram = float(m.get("size", 0)), float(m.get("size_vram", 0))
                if size <= 0:
                    return None
                return {"size_gb": size / 1e9, "vram_gb": vram / 1e9,
                        "gpu_pct": 100.0 * vram / size if size else 0.0}
    except Exception:
        pass
    return None


def list_ollama_models(host=None, timeout=2.0):
    host = host or _c("OLLAMA_HOST", "http://localhost:11434")
    try:
        u = urllib.parse.urlparse(host)
        conn = http.client.HTTPConnection(u.hostname or "localhost", u.port or 11434, timeout=timeout)
        conn.request("GET", "/api/tags")
        js = json.loads(conn.getresponse().read().decode("utf-8"))
        return [m["name"] for m in js.get("models", []) if "name" in m]
    except Exception:
        return []


class DaggerArbiter:
    """Expert (privileged) vs. LLM agent.  Returns the speed actually commanded."""

    def __init__(self, authority=None, dataset_path=None):
        self.authority = (authority or _c("AD_LLM_AUTHORITY", "dagger")).lower()
        self.threshold_kmh = float(_c("AD_DAGGER_THRESHOLD_KMH", 6.0))
        self.max_age = float(_c("AD_LLM_MAX_AGE_S", 2.0))
        self.dataset_path = dataset_path or _c("AD_DAGGER_DATASET", "dagger_dataset.jsonl")
        self.reset()

    def reset(self):
        self.decision = None
        self.takeovers = 0
        self.frames_llm = 0
        self.frames = 0
        self.last = {}

    def feed(self, dec):
        if dec is not None:
            self.decision = dec

    def arbitrate(self, plan, ego_kmh, scenario="", collect=True, sim_t=None):
        """plan = ad_types.PlanOut from the expert.  Returns (target_speed_mps, info dict)."""
        self.frames += 1
        exp_kmh = plan.target_speed * 3.6
        if plan.intent.startswith(("BYPASS", "EVADE", "LANE_CHANGE")):
            exp_act = "BYPASS_LEFT" if plan.intent.endswith("LEFT") else "BYPASS_RIGHT"
        else:
            exp_act = INTENT_TO_ACTION.get(plan.intent, "KEEP")
        info = {"source": "EXPERT", "llm_action": "-", "llm_speed_kmh": None, "expert_action": exp_act,
                "expert_speed_kmh": exp_kmh, "l1": None, "takeover": False, "age_s": None,
                "latency_ms": None, "failure": "LLM_NOT_READY"}
        dec = self.decision
        if self.authority == "expert" or dec is None:
            self.last = info
            return plan.target_speed, info
        age = time.monotonic() - dec.get("t_done", 0.0)
        meta = dec.get("meta") or {}
        if sim_t is not None and meta.get("sim_t") is not None:
            # age of the SCENE the LLM saw, in simulation seconds (correct when the sim runs
            # faster or slower than real time)
            age = sim_t - meta["sim_t"]
        info.update(llm_action=dec["action"], llm_speed_kmh=dec["target_speed_kmh"], age_s=age,
                    latency_ms=dec.get("latency_ms"))
        if age > self.max_age:
            info["failure"] = "FAIL_LATENCY"
            self.last = info
            return plan.target_speed, info
        l_kmh = dec["target_speed_kmh"]
        l1 = abs(l_kmh - exp_kmh)
        info["l1"] = l1
        unsafe = l_kmh > exp_kmh + self.threshold_kmh
        false_stop = l_kmh < 1.0 and exp_kmh > 8.0
        missed_yield = exp_act in ("STOP", "YIELD") and l_kmh > 5.0 and exp_kmh < 1.0
        info["failure"] = ("FAIL_UNSAFE_SPEED" if unsafe or missed_yield else
                           "FAIL_FALSE_STOP" if false_stop else
                           "FAIL_MISSED_OVERTAKE" if exp_act.startswith("BYPASS") and dec["action"] in ("STOP", "YIELD")
                           else "NONE")
        takeover = l1 > self.threshold_kmh or missed_yield or false_stop
        if self.authority == "advisory":
            # LLM may only lower the speed, never below what the expert considers necessary to move on
            v = min(exp_kmh, max(l_kmh, exp_kmh - self.threshold_kmh))
            info.update(source="LLM(advisory)", takeover=takeover)
        elif takeover:
            v = exp_kmh
            info.update(source="EXPERT(takeover)", takeover=True)
            self.takeovers += 1
            if collect:
                self._aggregate(dec, plan, scenario, exp_act, exp_kmh)
        else:
            v = min(l_kmh, exp_kmh)           # LLM drives inside the expert's safety envelope
            info.update(source="LLM")
            self.frames_llm += 1
        self.last = info
        return v / 3.6, info

    def _aggregate(self, dec, plan, scenario, exp_act, exp_kmh):
        if not self.dataset_path or dec.get("_logged"):
            return
        dec["_logged"] = True          # one sample per LLM reply
        meta = dec.get("meta") or {}
        label = {"action": meta.get("action", exp_act),
                 "target_speed": round(float(meta.get("speed_kmh", exp_kmh)), 1),
                 "reason": str(meta.get("reason", plan.reason))[:80]}
        if abs(label["target_speed"] - dec["target_speed_kmh"]) <= self.threshold_kmh \
                and label["action"] == dec["action"]:
            return                     # the LLM agreed with the expert on ITS input scene
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "time": round(time.time(), 2), "scenario": scenario, "model": dec.get("model"),
                    "system": SYSTEM_PROMPT, "prompt": dec.get("prompt"),
                    "llm": {"action": dec["action"], "target_speed": dec["target_speed_kmh"],
                            "reason": dec.get("reason", "")},
                    "label": label,
                }) + "\n")
        except Exception:
            pass
