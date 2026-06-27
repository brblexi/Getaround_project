---
title: GetAround Delay Analysis
emoji: 🚗
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GetAround — Delay Analysis Dashboard

Interactive dashboard quantifying the impact of late car returns on the next
rental, and simulating a **minimum-delay-between-rentals** threshold and scope
(all cars vs Connect-only).

Built with Streamlit + Plotly, served via Docker on Hugging Face Spaces.

## What it shows

1. **How often cars are returned late** — distribution and severity of checkout delays.
2. **Impact on the next driver** — impacted rate by check-in type and cancellation lift.
3. **Threshold simulator** — live trade-off between problems solved and rentals blocked.
4. **Reading of the data** — a recommended starting range to A/B test.

## Run locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker (mirrors the Space):

```bash
docker build -t getaround-dashboard .
docker run -p 7860:7860 getaround-dashboard
# open http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a new Space → **SDK: Docker**.
2. Push the contents of this `dashboard/` folder to the Space repo (this README,
   with its YAML header, must sit at the Space root).
3. The Space builds the Dockerfile and serves the app on port 7860.
