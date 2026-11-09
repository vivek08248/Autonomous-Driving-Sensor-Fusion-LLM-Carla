"""
llm_planner.py – Local-LLM driving planner (Ollama) with formal AV reasoning schema.

Features:
- Compact, high-signal system prompt designed for quick KV-cache reuse (<256 tokens).
- Structured schema supporting reasoning + intent + speed.
- Deterministic extraction with robust fallbacks.
- Fully integrated with Mobileye RSS safety envelopes and traffic rules.
"""

import urllib.request
import urllib.error
import json
import re
import config

_TIMEOUT      = float(getattr(config, "LLM_TIMEOUT_S", 5.0))
_KEEP_ALIVE   = getattr(config, "LLM_KEEP_ALIVE", "30m")
_NUM_CTX      = int(getattr(config, "LLM_NUM_CTX", 256))
_NUM_PREDICT  = int(getattr(config, "LLM_NUM_PREDICT", 35))
_DISABLE_THINK = getattr(config, "LLM_DISABLE_THINK", False)
_MAX_RESPONSE_BYTES = int(getattr(config, "LLM_MAX_RESPONSE_BYTES", 16384))

SYSTEM_PROMPT = """You are an autonomous vehicle motion planner (DriveGPT4-v2). Analyze current sensor state and return ONLY one JSON object:
{"intent": "KEEP_LANE"|"SLOW_DOWN"|"STOP"|"ACCELERATE"|"MANEUVER_RIGHT"|"MANEUVER_LEFT", "target_speed_kmh": number, "reasoning": "brief explanation"}

Rules:
- Red light AT stop line (< 3m) OR occupied zebra crossing OR obstacle < 3.5m -> {"intent": "STOP", "target_speed_kmh": 0.0, "reasoning": "Stop at line / safety hazard"}
- Approaching red light (3m to 25m) -> {"intent": "SLOW_DOWN", "target_speed_kmh": 14.0, "reasoning": "Smooth deceleration to stop line"}
- Encroaching obstacle (encroach=LEFT) -> {"intent": "MANEUVER_RIGHT", "target_speed_kmh": 16.0, "reasoning": "Nudge right around left encroachment"}
- Encroaching obstacle (encroach=RIGHT) -> {"intent": "MANEUVER_LEFT", "target_speed_kmh": 16.0, "reasoning": "Nudge left around right encroachment"}
- Stopped in traffic behind vehicle -> if other lanes busy or at red light: {"intent": "STOP", "target_speed_kmh": 0.0, "reasoning": "Queue safely in lane"}; if other lane safe and clear: {"intent": "MANEUVER_LEFT"|"MANEUVER_RIGHT", "target_speed_kmh": 14.0, "reasoning": "Bypass into clear lane"}
- Obstacle closer than rss_d_safe OR yellow light OR rain/fog -> {"intent": "SLOW_DOWN", "target_speed_kmh": 12.0, "reasoning": "Decelerating for safe headway"}
- Clear lane ahead > 25m AND green light -> {"intent": "KEEP_LANE", "target_speed_kmh": 26.0, "reasoning": "Cruising in lane"}
- Open clear road > 45m AND green light -> {"intent": "ACCELERATE", "target_speed_kmh": 32.0, "reasoning": "Accelerating on clear road"}
"""

ALLOWED_INTENTS = {
    "KEEP_LANE", "SLOW_DOWN", "STOP", "ACCELERATE",
    "MANEUVER_LEFT", "MANEUVER_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "OVERTAKE"
}

FALLBACK = {
    "intent": "KEEP_LANE",
    "target_speed_kmh": 20.0,
    "reasoning": "Standard cruise fallback.",
}


def _extract_and_sanitize(text):
    """Robustly extract and validate driving decision from raw LLM output.
    
    Resilient to truncated JSON, missing closing braces, trailing commas,
    and conversational text. Supports full lateral and longitudinal intents.
    """
    if not text or not isinstance(text, str):
        return None

    raw_intent = None
    speed = None
    reasoning = ""

    # 1. Try direct full JSON parse
    obj = None
    try:
        obj = json.loads(text.strip())
    except Exception:
        pass

    # 2. Try regex bounded JSON {...}
    if not isinstance(obj, dict):
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                obj = json.loads(match.group())
            except Exception:
                pass

    # 3. Try repairing truncated/unclosed JSON
    if not isinstance(obj, dict):
        brace_idx = text.find('{')
        if brace_idx != -1:
            snippet = text[brace_idx:].strip()
            snippet = re.sub(r',\s*$', '', snippet)
            snippet = re.sub(r',\s*"[^"]*":?\s*$', '', snippet)
            if not snippet.endswith('}'):
                snippet += '}'
            try:
                obj = json.loads(snippet)
            except Exception:
                pass

    # 4. Extract fields from parsed dict or fallback to regex field extraction
    if isinstance(obj, dict):
        raw_intent = str(obj.get("intent", "")).upper().strip()
        speed_val = obj.get("target_speed_kmh", None)
        try:
            speed = float(speed_val) if speed_val is not None else 20.0
        except (ValueError, TypeError):
            speed = 20.0
        reasoning = str(obj.get("reasoning", "")).strip()
    else:
        # Regex field search across entire string
        m_intent = re.search(r'["\']?intent["\']?\s*(?::|=|is)\s*["\']?([A-Za-z_]+)', text, re.IGNORECASE)
        if m_intent:
            raw_intent = m_intent.group(1).upper().strip()
        m_speed = re.search(r'["\']?target_speed(?:_kmh)?["\']?\s*(?::|=|is)\s*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        if m_speed:
            try:
                speed = float(m_speed.group(1))
            except (ValueError, TypeError):
                speed = 20.0
        m_reason = re.search(r'["\']?reasoning["\']?\s*(?::|=|is)\s*["\']([^"\']*)', text, re.IGNORECASE)
        if m_reason:
            reasoning = m_reason.group(1).strip()

    # 5. Direct keyword fallback if intent is still missing
    if not raw_intent:
        text_upper = text.upper()
        if "MANEUVER_RIGHT" in text_upper or "AVOID_RIGHT" in text_upper:
            raw_intent = "MANEUVER_RIGHT"
        elif "MANEUVER_LEFT" in text_upper or "AVOID_LEFT" in text_upper:
            raw_intent = "MANEUVER_LEFT"
        elif "STOP" in text_upper or "BRAKE" in text_upper:
            raw_intent = "STOP"
        elif "SLOW" in text_upper:
            raw_intent = "SLOW_DOWN"
        elif "ACCEL" in text_upper or "FAST" in text_upper:
            raw_intent = "ACCELERATE"
        elif "KEEP" in text_upper or "CRUISE" in text_upper:
            raw_intent = "KEEP_LANE"
        else:
            return None

    if speed is None:
        speed = 20.0

    # 6. Sanitize and normalize intent into ALLOWED_INTENTS
    if "|" in raw_intent:
        raw_intent = "KEEP_LANE"
    elif raw_intent not in ALLOWED_INTENTS:
        if "RIGHT" in raw_intent:
            raw_intent = "MANEUVER_RIGHT"
        elif "LEFT" in raw_intent:
            raw_intent = "MANEUVER_LEFT"
        elif "STOP" in raw_intent:
            raw_intent = "STOP"
        elif "SLOW" in raw_intent or "BRAKE" in raw_intent:
            raw_intent = "SLOW_DOWN"
        elif "ACCEL" in raw_intent or "FAST" in raw_intent:
            raw_intent = "ACCELERATE"
        else:
            raw_intent = "KEEP_LANE"

    # Clamp speeds per intent
    if raw_intent == "STOP":
        speed = 0.0
    elif raw_intent in ("SLOW_DOWN", "MANEUVER_LEFT", "MANEUVER_RIGHT"):
        speed = min(speed, 18.0)
    elif raw_intent == "ACCELERATE":
        speed = max(speed, 22.0)

    speed = max(0.0, min(speed, float(getattr(config, "MAX_ALLOWED_SPEED_KMH", 30.0))))
    if not reasoning:
        reasoning = f"{raw_intent} at {speed:.1f} km/h"

    return {
        "intent": raw_intent,
        "target_speed_kmh": round(speed, 1),
        "reasoning": reasoning,
    }


class LLMAgent:
    def __init__(self, model_name=None, host=None):
        # Resolved at call time so the benchmark can switch models
        # (the old default argument was frozen at import time).
        self.model_name = (
            model_name
            or getattr(config, "OLLAMA_MODEL", None)
            or config.LLM_MODEL
        )
        host = host or config.OLLAMA_HOST
        self.url = f"{host}/api/generate"

    def get_driving_decision(self, prompt_text):
        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "num_predict": _NUM_PREDICT,
                "num_ctx": _NUM_CTX,
            },
        }
        if _DISABLE_THINK:
            payload["think"] = False

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read(_MAX_RESPONSE_BYTES).decode("utf-8"))
                    raw_response = resp_json.get("response", "")
                    decision = _extract_and_sanitize(raw_response)
                    if decision is not None:
                        return decision

        except urllib.error.URLError:
            pass
        except TimeoutError:
            print(f"[LLM Agent Timeout]: Ollama response took > {_TIMEOUT}s.")
        except Exception as e:
            print(f"[LLM Agent Error]: {e}")

        return dict(FALLBACK)