# AI Robot Monitoring Dashboard

A single-file Streamlit demo for predictive robot health monitoring and alerts.

## Overview

This demo simulates a fleet of industrial robots and displays:
- live cluster KPIs
- robot health overview cards
- selected robot trending metrics and predictions
- active alerts and predictive alerts
- parameter correlation heatmap

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Start the dashboard with:

```bash
streamlit run app.py
```

## Deploy

This app is ready for online deployment. Recommended options:

1. **Streamlit Community Cloud**
   - Push this project to GitHub.
   - Create a new app from the repo.
   - Use `app.py` as the main file.
   - `requirements.txt` and `runtime.txt` are already included.

2. **Render**
   - Create a new Web Service.
   - Use `web: streamlit run app.py --server.port $PORT`.
   - Select `python-3.11.18` from `runtime.txt`.

3. **Heroku**
   - Deploy from GitHub or via the Heroku CLI.
   - The `Procfile` is included for correct startup.

## Notes

- The app uses simulated robot telemetry and simple linear trend prediction.
- Update interval and prediction horizon are controllable from the sidebar.
- The selected robot panel shows per-unit predictions and trend charts.
