import json
import numpy as np
import pandas as pd
import io
import os
import re
from datetime import date
from typing import Dict, Any, Optional

import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



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
# Local lightweight ML classifier (no external APIs)
# -----------------------------
TRAIN_DATA = [
    ("Delivered today. POD confirmed. No issues.", "On-time"),
    ("Arrived on schedule. Signed proof of delivery uploaded.", "On-time"),
    ("On track; ETA unchanged.", "On-time"),

    ("Pickup rescheduled for tomorrow due to weather.", "Minor delay"),
    ("Late pickup. ETA slips by 1 day.", "Minor delay"),
    ("Appointment moved to next day; slight delay expected.", "Minor delay"),

    ("Customs hold due to missing documents; clearance may take several days.", "Major delay"),
    ("Port congestion and rolled booking; new ETA pending vessel schedule.", "Major delay"),
    ("Mechanical breakdown; freight delayed 4 days.", "Major delay"),

    ("Port strike; terminal shutdown until further notice.", "Disruption likely"),
    ("Shipment missing; investigation opened.", "Disruption likely"),
    ("Facility fire reported; operations suspended.", "Disruption likely"),
]

@st.cache_resource
def get_local_model():
    X = [t for t, y in TRAIN_DATA]
    y = [y for t, y in TRAIN_DATA]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y)
    feature_names = vec.get_feature_names_out()
    return vec, clf, feature_names
    
def top_keywords_for_prediction(vec, clf, feature_names, Xv_row, predicted_label: str, top_k: int = 10):
    """
    Returns the top positive contributing keywords/phrases for the predicted class.
    Works for TF-IDF + Logistic Regression.
    """
    # For multi-class LogisticRegression, coef_ is shape (n_classes, n_features)
    class_index = list(clf.classes_).index(predicted_label)
    coefs = clf.coef_[class_index]  # shape (n_features,)

    # Xv_row is a sparse row vector; convert to COO to iterate non-zeros
    row = Xv_row.tocoo()
    contributions = []

    for j, v in zip(row.col, row.data):
        score = v * coefs[j]  # tfidf_value * weight
        if score > 0:
            contributions.append((feature_names[j], float(score)))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_k]
def local_predict(text: str) -> Dict[str, Any]:
    vec, clf, feature_names = get_local_model()

    Xv = vec.transform([text])              # shape (1, n_features)
    probs = clf.predict_proba(Xv)[0]        # array of probabilities
    classes = clf.classes_                  # class labels

    best_i = int(np.argmax(probs))
    label = str(classes[best_i])
    confidence = float(probs[best_i])

    # Build probability breakdown
    prob_breakdown = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
    top_terms = top_keywords_for_prediction(vec, clf, feature_names, Xv, label, top_k=10)
    # Map label to action
    action_map = {
        "On-time": "monitor",
        "Minor delay": "monitor",
        "Major delay": "expedite",
        "Disruption likely": "escalate",
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_breakdown,
        "top_keywords": top_terms,
        "recommended_action": action_map.get(label, "monitor"),
        "rationale": "Predicted by a lightweight text classifier trained on example exception notes.",
        "assumptions": ["Training set is small and illustrative; expand with real historical notes for production."]
    }


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

def show_confidence_indicator(conf: float):
    if conf >= 0.75:
        st.success(f"Confidence: {conf:.2f} (High)")
    elif conf >= 0.50:
        st.warning(f"Confidence: {conf:.2f} (Medium)")
    else:
        st.error(f"Confidence: {conf:.2f} (Low)")
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Shipment Delay Classification Tool", layout="wide")
st.title("Shipment Delay Classification Tool")
st.caption("AI-enabled shipment exception triage: classify severity + recommend next action (free-tier stack).")


with st.sidebar:
    st.header("Model / Settings")
    st.write("Classifier: Local TF-IDF + Logistic Regression (no external API)")
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

    try:
        result = local_predict(exception_text)

        st.success("AI classification complete.")
        st.subheader("Result")
        st.write(f"**Label:** {result['label']}")
        show_confidence_indicator(result["confidence"])
        
        st.write("**Probability breakdown:**")
        st.json(result["probabilities"])
        
        st.write("**Why this classification? (Top keywords)**")
        if result.get("top_keywords"):
            st.table(pd.DataFrame(result["top_keywords"], columns=["keyword/phrase", "impact_score"]))
        else:
            st.write("No strong keyword signals found in this note.")
        st.write(f"**Recommended action:** {result['recommended_action']}")
        st.write(f"**Rationale:** {result['rationale']}")
        
        if result["assumptions"]:
            st.write("**Assumptions:**")
            st.write(result["assumptions"])

        st.caption("Note: Local model trained on example cases. Expand training data for production.")

    except Exception as e:
        st.warning(f"Model failed, using fallback rules. Details: {repr(e)}")
        fb = fallback_classifier(exception_text)
        st.subheader("Fallback Result (Rules-Based)")
        st.write(f"**Label:** {fb['label']}")
        st.write(f"**Confidence:** {fb['confidence']:.2f}")
        st.write(f"**Recommended action:** {fb['recommended_action']}")

# -----------------------------
# Batch classify (CSV upload)
# -----------------------------
st.divider()
st.header("Batch classify (CSV upload)")
st.write("Upload a CSV with a column named **exception_note** (required). Optional columns: lane, mode, promised_date.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "exception_note" not in df.columns:
            st.error("CSV must contain a column named: exception_note")
            st.stop()

        # Fill missing optional columns if not provided
        if "lane" not in df.columns:
            df["lane"] = ""
        if "mode" not in df.columns:
            df["mode"] = ""
        if "promised_date" not in df.columns:
            df["promised_date"] = ""

        results = []
        for _, row in df.iterrows():
            note = str(row["exception_note"])
            pred = local_predict(note)

            results.append({
                "exception_note": note,
                "label": pred["label"],
                "confidence": round(pred["confidence"], 4),
                "recommended_action": pred["recommended_action"],
                "prob_on_time": round(pred["probabilities"].get("On-time", 0.0), 4),
                "prob_minor": round(pred["probabilities"].get("Minor delay", 0.0), 4),
                "prob_major": round(pred["probabilities"].get("Major delay", 0.0), 4),
                "prob_disruption": round(pred["probabilities"].get("Disruption likely", 0.0), 4),
            })

        out_df = pd.DataFrame(results)

        st.subheader("Batch results preview")
        st.dataframe(out_df, use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="shipment_delay_batch_results.csv",
            mime="text/csv"
        )

    except Exception as e:
            st.error(f"Batch processing failed: {repr(e)}")
