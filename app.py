import json
import os
import re
from datetime import date
from typing import Dict, Any, Optional

import requests
import streamlit as st
from huggingface_hub import InferenceClient


# -----------------------------
# Config
# -----------------------------
LABELS = ["On-time", "Minor delay", "Major delay", "Disruption likely"]
ACTIONS = ["monitor", "notify_customer", "expedite", "escalate"]


# -----------------------------
# Helper: simple fallback rules
# -----------------------------
def fallback_classifier(text: str) -> Dict[str, Any]:
    t = text.lower()

    disruption_keywords = [
        "strike", "closure", "shut down", "shutdown", "port closed", "riot", "earthquake",
        "fire", "lost", "missing", "stolen", "bankruptcy", "insolvency", "embargo"
    ]
    major_keywords = [
        "customs hold", "held in customs", "port congestion", "backlog", "capacity",
        "mechanical", "breakdown", "rolled", "no truck", "appointment missed", "missed appointment",
        "delayed 3", "delayed 4", "delayed 5", "delayed 6", "delayed 7"
    ]
    minor_keywords = [
        "rescheduled", "late pickup", "late pick-up", "eta tomorrow", "eta next day",
        "delayed 1", "delayed 2", "one day late", "two days late"
    ]

    if any(k in t for k in disruption_keywords):
        return {"label": "Disruption likely", "confidence": 0.55, "recommended_action": "escalate"}
    if any(k in t for k in major_keywords):
        return {"label": "Major delay", "confidence": 0.55, "recommended_action": "expedite"}
    if any(k in t for k in minor_keywords):
        return {"label": "Minor delay", "confidence": 0.55, "recommended_action": "monitor"}
    return {"label": "Minor delay", "confidence": 0.40, "recommended_action": "monitor"}


# -----------------------------
# Hugging Face Inference call
# -----------------------------
def hf_generate(model: str, prompt: str, hf_token: str, timeout_s: int = 45) -> str:
    # IMPORTANT: router endpoint must include /hf-inference/
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

    # Helpful error details (shows HF response body)
    if r.status_code >= 400:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")

    data = r.json()

    # Most common: list[{"generated_text": "..."}]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]

    # Sometimes: {"generated_text": "..."}
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]

    return json.dumps(data)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to find a JSON object in the model output.
    """
    # common: model outputs ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # otherwise find first {...}
    curly = re.search(r"(\{.*\})", text, re.DOTALL)
    if curly:
        try:
            return json.loads(curly.group(1))
        except Exception:
            return None

    return None


def normalize_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    label = str(obj.get("label", "")).strip()
    if label not in LABELS:
        label = "Minor delay"

    confidence = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    recommended_action = str(obj.get("recommended_action", "monitor")).strip()
    if recommended_action not in ACTIONS:
        recommended_action = "monitor"

    rationale = str(obj.get("rationale", "")).strip()
    if not rationale:
        rationale = "No rationale provided."

    assumptions = obj.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = [str(assumptions)]

    return {
        "label": label,
        "confidence": confidence,
        "recommended_action": recommended_action,
        "rationale": rationale,
        "assumptions": assumptions
    }


def build_prompt(exception_text: str, mode: str, lane: str, promised_date: str) -> str:
    return f"""
You are a supply chain shipment exception triage assistant.
Your job is to classify delay severity from shipment exception notes and recommend an action.

Return ONLY valid JSON with this schema:
{{
  "label": "On-time" | "Minor delay" | "Major delay" | "Disruption likely",
  "confidence": number between 0 and 1,
  "rationale": "1-2 short sentences",
  "recommended_action": "monitor" | "notify_customer" | "expedite" | "escalate",
  "assumptions": ["list", "of", "assumptions"]
}}

Rules:
- If the note suggests indefinite timing, shutdown/strike, missing freight, security event, or insolvency -> "Disruption likely"
- If delay likely 3-7+ days or requires major intervention (customs hold, port congestion, capacity failure) -> "Major delay"
- If delay likely 1-2 days or minor reschedule -> "Minor delay"
- If explicitly delivered/on schedule -> "On-time"
- If uncertain, choose best label but LOWER confidence and list assumptions.

Shipment context:
- mode: {mode}
- lane: {lane}
- promised_date: {promised_date}

Exception note:
{exception_text}
""".strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Shipment Delay Classification Tool", layout="wide")
st.title("Shipment Delay Classification Tool")
st.caption("AI-enabled shipment exception triage: classify severity + recommend next action (free-tier stack).")
# --- Load Hugging Face token (Streamlit secrets first, env var second) ---
hf_token = ""
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = os.getenv("HF_TOKEN", "")

with st.sidebar:
    st.header("Model / Settings")

    # Shows whether Streamlit can see your token (does not reveal token)
    token_present = bool(hf_token and str(hf_token).strip())
    st.write("HF_TOKEN detected:", "✅ Yes" if token_present else "❌ No")

    # Model selector
    model_name = st.text_input(
        "Hugging Face model (text-generation)",
        value="google/flan-t5-base"
    )

    st.write("Tip: if this model errors or is slow, switch to a smaller instruction model on HF.")
    st.divider()

    st.header("Examples")
    example = st.selectbox(
        "Load an example",
        [
            "—",
            "Port congestion: rolled booking",
            "Customs hold: paperwork missing",
            "Weather delay: snow, 1-day slip",
            "Delivered: POD confirmed",
            "Strike: terminal shutdown"
        ],
    )

# Set default text based on example
default_text = ""
if example == "Port congestion: rolled booking":
    default_text = "Carrier advises booking was rolled due to port congestion; new ETA unknown pending vessel schedule."
elif example == "Customs hold: paperwork missing":
    default_text = "Shipment held in customs due to missing commercial invoice. Broker requesting documents; clearance may take several days."
elif example == "Weather delay: snow, 1-day slip":
    default_text = "Pickup delayed due to snow. Driver rescheduled for tomorrow; ETA slips by 1 day."
elif example == "Delivered: POD confirmed":
    default_text = "Delivered today 10:14. POD uploaded. No issues reported."
elif example == "Strike: terminal shutdown":
    default_text = "Port labor strike announced; terminal operations suspended until further notice."

# Main page inputs
col1, col2 = st.columns(2)

with col1:
    exception_text = st.text_area("Exception note / carrier update", value=default_text, height=220)
    mode = st.selectbox("Mode", ["Ocean", "Air", "Truck", "Rail", "Parcel"])

with col2:
    lane = st.text_input("Lane (e.g., CN→LAX, NJ→TX)", value="CN→LAX")
    promised_date = st.date_input("Promised delivery date", value=date.today()).isoformat()
    st.write("")

classify = st.button("Classify delay")





if classify:
    if not exception_text.strip():
        st.error("Please paste an exception note.")
        st.stop()

    # Try AI classification if token exists
    if hf_token:
        prompt = build_prompt(exception_text, mode, lane, promised_date)
        try:
            raw = hf_generate(model_name, prompt, hf_token)
            parsed = extract_json(raw)

            # If JSON is bad, retry once with a stricter instruction
            if parsed is None:
                strict_prompt = prompt + "\n\nIMPORTANT: Output must be ONLY JSON. No prose. No markdown."
                raw2 = hf_generate(model_name, strict_prompt, hf_token)
                parsed = extract_json(raw2)

            if parsed is None:
                raise ValueError("Model did not return valid JSON.")

            result = normalize_result(parsed)

            st.success("AI classification complete.")
            st.subheader("Result")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Confidence:** {result['confidence']:.2f}")
            st.write(f"**Recommended action:** {result['recommended_action']}")
            st.write(f"**Rationale:** {result['rationale']}")
            if result["assumptions"]:
                st.write("**Assumptions:**")
                st.write(result["assumptions"])

            st.caption("Note: Model output is validated and normalized. If the model fails, tool falls back to rules.")

        except Exception as e:
            st.warning(f"AI call failed, using fallback rules. Details: {repr(e)}")
            fb = fallback_classifier(exception_text)
            st.subheader("Fallback Result (Rules-Based)")
            st.write(f"**Label:** {fb['label']}")
            st.write(f"**Confidence:** {fb['confidence']:.2f}")
            st.write(f"**Recommended action:** {fb['recommended_action']}")
    else:
        st.warning("No HF_TOKEN found. Add it to Streamlit secrets for AI mode. Using fallback rules for now.")
        fb = fallback_classifier(exception_text)
        st.subheader("Fallback Result (Rules-Based)")
        st.write(f"**Label:** {fb['label']}")
        st.write(f"**Confidence:** {fb['confidence']:.2f}")
        st.write(f"**Recommended action:** {fb['recommended_action']}")
