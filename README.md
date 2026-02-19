# AI-Based Shipment Exception Triage Tool

Shipment Delay Classification Tool (Streamlit + Hugging Face free inference).

## What it does
Paste a carrier/shipment exception note and get:
- Delay severity classification (On-time / Minor / Major / Disruption likely)
- Confidence score
- Recommended action (monitor / notify_customer / expedite / escalate)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
