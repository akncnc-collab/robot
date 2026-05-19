import time
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# Simple AI-based Robot Monitoring Demo (single-file)
# Minimal deps: streamlit, plotly, numpy


NUM_ROBOTS = 20
HISTORY_LENGTH = 60
PROBLEM_ROBOTS = {"R007", "R012", "R015"}
AI_RISK_RULES = {
    "temperature": {"threshold": 80, "label": "Temperature"},
    "vibration": {"threshold": 70, "label": "Vibration"},
    "torque": {"threshold": 85, "label": "Torque"},
}


def init_session():
    if "robots" not in st.session_state:
        robots = []
        for i in range(1, NUM_ROBOTS + 1):
            robot_id = f"R{i:03d}"
            base_temp = random.uniform(55, 75)
            base_torque = random.uniform(45, 70)
            base_speed = random.uniform(900, 1400)
            base_vib = random.uniform(15, 40)
            runtime_hours = random.uniform(100, 2000)
            r = {
                "robot_id": robot_id,
                "id": robot_id,
                "temperature": base_temp,
                "torque": base_torque,
                "speed": base_speed,
                "vibration": base_vib,
                "runtime_hours": runtime_hours,
                "status": "Healthy",
                "health_score": 100,
                "risk_score": 0,
                "predicted_failure_probability": 5,
                "alert_summary": "None",
                "alarms": 0,
                "history": {
                    "ts": deque(maxlen=HISTORY_LENGTH),
                    "temperature": deque(maxlen=HISTORY_LENGTH),
                    "torque": deque(maxlen=HISTORY_LENGTH),
                    "speed": deque(maxlen=HISTORY_LENGTH),
                    "vibration": deque(maxlen=HISTORY_LENGTH),
                },
            }
            # Fill history with initial stable samples
            now = datetime.now()
            for j in range(HISTORY_LENGTH):
                t = now
                r["history"]["ts"].append(t)
                r["history"]["temperature"].append(base_temp + random.uniform(-1, 1))
                r["history"]["torque"].append(base_torque + random.uniform(-1.5, 1.5))
                r["history"]["speed"].append(base_speed + random.uniform(-10, 10))
                r["history"]["vibration"].append(base_vib + random.uniform(-1, 1))
            r["status"] = ai_health_check(r)
            r["health_score"] = robot_health_score(r)
            r["risk_score"] = robot_risk_score(r)
            r["predicted_failure_probability"] = predicted_failure_probability(r)
            r["alert_summary"] = robot_alert_summary(r)
            robots.append(r)
        st.session_state.robots = robots
        st.session_state.running = False
        st.session_state.alarm_count = 0
        st.session_state.downtime_hours = 0.0
        st.session_state.avail_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.health_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.ts_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.pred_failures_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.active_alerts_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.downtime_pct_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.interval = 2
        st.session_state.prediction_horizon = 60
    if "robots" in st.session_state:
        for r in st.session_state.robots:
            r["status"] = ai_health_check(r)
            r["health_score"] = robot_health_score(r)
            r["risk_score"] = robot_risk_score(r)
            r["predicted_failure_probability"] = predicted_failure_probability(r)
            r["alert_summary"] = robot_alert_summary(r)


def ai_health_check(robot):
    breaches = robot_risk_breaches(robot)
    if len(breaches) >= 2:
        return "Critical"
    if len(breaches) == 1:
        return "Warning"
    return "Healthy"


def robot_risk_breaches(robot):
    breaches = []
    for metric, rule in AI_RISK_RULES.items():
        if robot[metric] > rule["threshold"]:
            breaches.append(
                {
                    "metric": metric,
                    "label": rule["label"],
                    "value": robot[metric],
                    "threshold": rule["threshold"],
                }
            )
    return breaches


def robot_alert_summary(robot):
    breaches = robot_risk_breaches(robot)
    if not breaches:
        return "None"
    return ", ".join(f"{b['label']} > {b['threshold']}" for b in breaches)


def robot_health_score(robot):
    # Estimate a health score from metrics for dashboard display
    temp = robot["temperature"]
    torque = robot["torque"]
    vib = robot["vibration"]
    score = 100.0
    score -= max(0.0, (temp - 60.0) * 0.8)
    score -= max(0.0, (torque - 55.0) * 0.7)
    score -= max(0.0, (vib - 30.0) * 0.9)
    if temp > 80:
        score -= (temp - 80) * 1.5
    if vib > 70:
        score -= (vib - 70) * 1.8
    if torque > 85:
        score -= (torque - 85) * 1.2
    score -= max(0, len(robot_risk_breaches(robot)) - 1) * 12
    score = np.clip(score, 10.0, 100.0)
    return int(score)


def robot_risk_score(robot):
    score = 0.0
    score += min(40.0, max(0.0, (robot["temperature"] - 60.0) * 1.0))
    score += min(30.0, max(0.0, (robot["torque"] - 55.0) * 0.8))
    score += min(30.0, max(0.0, (robot["vibration"] - 30.0) * 0.9))
    breaches = robot_risk_breaches(robot)
    score += len(breaches) * 12.0
    if len(breaches) >= 2:
        score += 18.0
    return int(np.clip(score, 0.0, 100.0))


def predicted_failure_probability(robot):
    ts_list = list(robot["history"]["ts"])
    if len(ts_list) < 3:
        base = 10
        return int(np.clip(base + robot_risk_score(robot) * 0.4, 5, 95))
    t0 = ts_list[0]
    secs = np.array([(t - t0).total_seconds() for t in ts_list])
    prob = 10.0
    for metric, threshold, weight in [("temperature", 80, 1.1), ("vibration", 70, 1.2), ("torque", 85, 1.0)]:
        vals = np.array(list(robot["history"][metric]))
        if np.ptp(secs) > 0:
            try:
                slope, intercept = np.polyfit(secs, vals, 1)
            except Exception:
                slope = 0.0
            current = float(vals[-1])
            if current >= threshold:
                prob += 25 * weight
            elif slope > 0:
                time_to_threshold = (threshold - current) / slope if slope > 0 else np.inf
                if time_to_threshold < 60:
                    prob += 25 * weight
                elif time_to_threshold < 180:
                    prob += 18 * weight
                elif time_to_threshold < 600:
                    prob += 12 * weight
                elif time_to_threshold < 1800:
                    prob += 8 * weight
                else:
                    prob += 4 * weight
        else:
            if robot[metric] > threshold:
                prob += 20 * weight
    prob += robot_risk_score(robot) * 0.2
    return int(np.clip(prob, 5.0, 99.0))


def trend_arrow(current, previous, positive_good=True):
    if previous is None:
        return ""
    if current == previous:
        return "→"
    if current > previous:
        return "⬆️" if positive_good else "⬇️"
    return "⬇️" if positive_good else "⬆️"


def update_robots(interval_seconds=2.0):
    now = datetime.now()
    avg_temp = avg_torque = avg_speed = avg_vib = 0.0
    critical_count = warning_count = 0
    for r in st.session_state.robots:
        # Small drift plus occasional spikes
        r["temperature"] += random.uniform(-0.4, 0.8)
        r["torque"] += random.uniform(-0.6, 0.6)
        r["speed"] += random.uniform(-10, 10)
        r["vibration"] += random.uniform(-0.5, 0.6)

        # gradual degradation for problem robots
        if r["id"] in PROBLEM_ROBOTS:
            r["temperature"] += random.uniform(0.2, 0.8)
            r["torque"] += random.uniform(0.4, 1.2)
            r["vibration"] += random.uniform(0.2, 0.6)
            if random.random() < 0.08:
                r["temperature"] += random.uniform(0.5, 1.4)
            if random.random() < 0.06:
                r["torque"] += random.uniform(1.2, 3.5)
            if random.random() < 0.08:
                r["vibration"] += random.uniform(0.8, 2.0)

        if random.random() < 0.02:
            r["temperature"] += random.uniform(4, 10)  # rare spike
        if random.random() < 0.015:
            r["torque"] += random.uniform(8, 18)
        if random.random() < 0.02:
            r["vibration"] += random.uniform(4, 12)

        # clamp sensible industrial ranges
        r["temperature"] = float(np.clip(r["temperature"], 20, 200))
        r["torque"] = float(np.clip(r["torque"], 0, 200))
        r["speed"] = float(np.clip(r["speed"], 0, 5000))
        r["vibration"] = float(np.clip(r["vibration"], 0, 200))

        # runtime increments
        r["runtime_hours"] += interval_seconds / 3600.0

        # history append
        r["history"]["ts"].append(now)
        r["history"]["temperature"].append(r["temperature"])
        r["history"]["torque"].append(r["torque"])
        r["history"]["speed"].append(r["speed"])
        r["history"]["vibration"].append(r["vibration"])

        # AI logic
        new_status = ai_health_check(r)
        r["status"] = new_status
        r["health_score"] = robot_health_score(r)
        r["risk_score"] = robot_risk_score(r)
        r["predicted_failure_probability"] = predicted_failure_probability(r)
        r["alert_summary"] = robot_alert_summary(r)
        if new_status != "Healthy":
            r["alarms"] += 1
            st.session_state.alarm_count += 1

        if new_status == "Critical":
            critical_count += 1
            # simulate some downtime accumulation
            st.session_state.downtime_hours += (interval_seconds / 3600.0) * 0.05
        elif new_status == "Warning":
            warning_count += 1

        avg_temp += r["temperature"]
        avg_torque += r["torque"]
        avg_speed += r["speed"]
        avg_vib += r["vibration"]

    n = len(st.session_state.robots)
    avg_temp /= n
    avg_torque /= n
    avg_speed /= n
    avg_vib /= n

    # Availability % (simple approximation)
    availability = max(0.0, 100.0 - (st.session_state.downtime_hours / max(1.0, sum(r['runtime_hours'] for r in st.session_state.robots)) * 100.0))
    health_score = float(np.mean([r["health_score"] for r in st.session_state.robots]))

    st.session_state.avail_history.append(availability)
    st.session_state.health_history.append(health_score)
    st.session_state.ts_history.append(now)

    return {
        "avg_temp": avg_temp,
        "avg_torque": avg_torque,
        "avg_speed": avg_speed,
        "avg_vib": avg_vib,
        "availability": availability,
        "health_score": health_score,
        "critical_count": critical_count,
        "warning_count": warning_count,
    }


def predictive_assessment(horizon_seconds=60):
    """Predict next values for each robot using a simple linear fit on recent history.
    Returns list of predictive alerts as dicts: {id, metric, predicted, threshold, level}
    """
    alerts = []
    for r in st.session_state.robots:
        # worker timestamps and values
        ts_list = list(r["history"]["ts"])
        if len(ts_list) < 2:
            continue
        t0 = ts_list[0]
        secs = np.array([(t - t0).total_seconds() for t in ts_list])
        # for each metric, fit linear slope
        robot_alerts = []
        for metric, rule in AI_RISK_RULES.items():
            thresh = rule["threshold"]
            vals = np.array(list(r["history"][metric]))
            try:
                # linear fit (degree 1) on time vs value
                if len(secs) >= 2 and np.ptp(secs) > 0:
                    p = np.polyfit(secs, vals, 1)
                    slope = p[0]
                    intercept = p[1]
                    predict_t = secs[-1] + horizon_seconds
                    predicted = float(slope * predict_t + intercept)
                else:
                    predicted = float(vals[-1])
            except Exception:
                predicted = float(vals[-1])

            if predicted > thresh:
                robot_alerts.append({"id": r["id"], "metric": metric, "predicted": predicted, "threshold": thresh, "level": "Warning"})
        if len(robot_alerts) >= 2:
            for alert in robot_alerts:
                alert["level"] = "Critical"
        alerts.extend(robot_alerts)
    return alerts


def predict_robot_metrics(robot, horizon_seconds=60):
    ts_list = list(robot["history"]["ts"])
    if len(ts_list) < 2:
        return {}
    t0 = ts_list[0]
    secs = np.array([(t - t0).total_seconds() for t in ts_list])
    predictions = {}
    predicted_breaches = []
    for metric, rule in AI_RISK_RULES.items():
        thresh = rule["threshold"]
        vals = np.array(list(robot["history"][metric]))
        predicted = float(vals[-1])
        if len(secs) >= 2 and np.ptp(secs) > 0:
            try:
                p = np.polyfit(secs, vals, 1)
                slope = p[0]
                intercept = p[1]
                predicted = float(slope * (secs[-1] + horizon_seconds) + intercept)
            except Exception:
                predicted = float(vals[-1])
        level = "Warning" if predicted > thresh else "Healthy"
        if predicted > thresh:
            predicted_breaches.append(metric)
        predictions[metric] = {
            "predicted": predicted,
            "threshold": thresh,
            "level": level,
        }
    if len(predicted_breaches) >= 2:
        for metric in predicted_breaches:
            predictions[metric]["level"] = "Critical"
    return predictions


def style_plotly_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.92)",
        plot_bgcolor="rgba(17,24,39,0.95)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        title=dict(x=0.01, xanchor="left", font=dict(size=18, color="#f8fafc")),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", font=dict(color="#e2e8f0")),
        hovermode="x unified",
    )
    if hasattr(fig, "update_xaxes"):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False, showline=False, color="#cbd5e1")
    if hasattr(fig, "update_yaxes"):
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False, showline=False, color="#cbd5e1")
    return fig


def build_gauge(title, value, min_val=0, max_val=100, threshold=None, suffix="", subtext=""):
    steps = [
        dict(range=[min_val, max_val * 0.4], color="#7f1d1d"),
        dict(range=[max_val * 0.4, max_val * 0.75], color="#f59e0b"),
        dict(range=[max_val * 0.75, max_val], color="#22c55e"),
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number=dict(suffix=suffix, font=dict(size=28, color="#f8fafc")),
            delta=dict(reference=max_val, relative=False, position="top", font=dict(color="#94a3b8")),
            gauge=dict(
                axis=dict(range=[min_val, max_val], tickcolor="#cbd5e1"),
                bar=dict(color="#38bdf8"),
                steps=steps,
                threshold=dict(line=dict(color="#facc15", width=4), thickness=0.75, value=threshold if threshold is not None else max_val * 0.75),
            ),
            title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        )
    )
    if subtext:
        fig.add_annotation(x=0.5, y=-0.15, text=subtext, showarrow=False, font=dict(color="#94a3b8", size=12))
    style_plotly_chart(fig)
    return fig


def build_risk_ranking_chart(robots):
    sorted_robots = sorted(robots, key=lambda r: (r["health_score"], r["temperature"]))
    df = pd.DataFrame(
        {
            "Robot": [r["id"] for r in sorted_robots],
            "Health Score": [r["health_score"] for r in sorted_robots],
            "Status": [r["status"] for r in sorted_robots],
        }
    )
    fig = px.bar(
        df,
        x="Robot",
        y="Health Score",
        color="Status",
        color_discrete_map={"Healthy": "#22c55e", "Warning": "#f59e0b", "Critical": "#ef4444"},
        title="Robot Health Ranking",
    )
    fig.update_traces(marker_line_color="#0f172a", marker_line_width=1)
    style_plotly_chart(fig)
    return fig


def build_metric_trend_chart(ts, values, metric_name, y_label, color):
    fig = px.line(
        pd.DataFrame({"ts": ts, y_label: values}),
        x="ts",
        y=y_label,
        title=f"{metric_name} Trend",
    )
    fig.update_traces(line=dict(color=color, width=3), hovertemplate="%{x}<br>%{y:.2f}")
    style_plotly_chart(fig)
    return fig


def build_alarm_pie(robots):
    counts = {
        "Healthy": sum(1 for r in robots if r["status"] == "Healthy"),
        "Warning": sum(1 for r in robots if r["status"] == "Warning"),
        "Critical": sum(1 for r in robots if r["status"] == "Critical"),
    }
    df = pd.DataFrame({"Status": list(counts.keys()), "Count": list(counts.values())})
    fig = px.pie(df, names="Status", values="Count", title="Alarm / Status Distribution", hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444"]))
    style_plotly_chart(fig)
    return fig


def build_robot_health_bar(robots):
    sorted_robots = sorted(robots, key=lambda r: r["health_score"])
    df = pd.DataFrame(
        {
            "Robot": [r["id"] for r in sorted_robots],
            "Health Score": [r["health_score"] for r in sorted_robots],
            "Status": [r["status"] for r in sorted_robots],
        }
    )
    fig = px.bar(
        df,
        x="Robot",
        y="Health Score",
        color="Status",
        color_discrete_map={"Healthy": "#38bdf8", "Warning": "#f97316", "Critical": "#ef4444"},
        title="Robot Health Score Bar Chart",
    )
    fig.update_traces(marker_line_width=0)
    style_plotly_chart(fig)
    return fig


def build_downtime_trend(ts, downtime_history):
    fig = px.area(
        pd.DataFrame({"ts": ts, "Downtime %": downtime_history}),
        x="ts",
        y="Downtime %",
        title="Downtime Trend",
    )
    fig.update_traces(fill="tozeroy", line=dict(color="#f97316", width=2), hovertemplate="%{x}<br>%{y:.2f}%")
    style_plotly_chart(fig)
    return fig


def forecast_robot_metric(robot, metric, horizon_seconds=60, warning_threshold=None, critical_threshold=None, points=30):
    ts_list = list(robot["history"]["ts"])
    values = np.array(list(robot["history"][metric]))
    if len(ts_list) < 3 or np.ptp(values) == 0:
        return None
    t0 = ts_list[0]
    secs = np.array([(t - t0).total_seconds() for t in ts_list])
    if len(secs) < 2 or np.ptp(secs) == 0:
        return None
    try:
        slope, intercept = np.polyfit(secs, values, 1)
    except Exception:
        return None
    now_sec = secs[-1]
    future_secs = np.linspace(now_sec, now_sec + horizon_seconds, points)
    future_times = [t0 + timedelta(seconds=float(s)) for s in future_secs]
    predicted = intercept + slope * future_secs
    residuals = values - (slope * secs + intercept)
    sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
    band = np.clip(sigma * 1.8, 1.2, max(3.0, abs(predicted).max() * 0.05))
    upper = predicted + band
    lower = predicted - band
    crossing_time = None
    crossing_level = None
    if slope > 0:
        for future_time, future_value in zip(future_times, predicted):
            if critical_threshold is not None and future_value >= critical_threshold:
                crossing_time = future_time
                crossing_level = "Critical"
                break
            if warning_threshold is not None and future_value >= warning_threshold and crossing_level is None:
                crossing_time = future_time
                crossing_level = "Warning"
                break
    risk_level = "Healthy"
    if critical_threshold is not None and predicted[-1] >= critical_threshold:
        risk_level = "Critical"
    elif warning_threshold is not None and predicted[-1] >= warning_threshold:
        risk_level = "Warning"
    if crossing_level == "Critical":
        risk_level = "Critical"
    elif crossing_level == "Warning" and risk_level != "Critical":
        risk_level = "Warning"
    confidence_pct = int(np.clip(90 - sigma * 6 + min(8, len(values) // 4), 35, 96))
    predicted_failure_time = crossing_time.strftime("%I:%M:%S %p") if crossing_time else "No breach predicted"
    return {
        "metric": metric,
        "forecast_times": future_times,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "risk_level": risk_level,
        "confidence_pct": confidence_pct,
        "predicted_failure_time": predicted_failure_time,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "history_ts": ts_list,
        "history_values": values.tolist(),
    }


def build_future_prediction_chart(forecast, metric_name, units, color):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["history_ts"],
            y=forecast["history_values"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=4, color="#38bdf8"),
            hovertemplate="%{x}<br>Actual: %{y:.2f}"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["forecast_times"],
            y=forecast["predicted"],
            mode="lines",
            name="Predicted",
            line=dict(color=color, width=3, dash="dash"),
            hovertemplate="%{x}<br>Forecast: %{y:.2f}"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast["forecast_times"]) + list(reversed(forecast["forecast_times"])),
            y=list(forecast["upper"]) + list(reversed(forecast["lower"])),
            fill="toself",
            fillcolor="rgba(59,130,246,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Confidence band",
        )
    )
    if forecast["warning_threshold"] is not None:
        fig.add_hline(
            y=forecast["warning_threshold"],
            line=dict(color="#f59e0b", dash="dot"),
            annotation_text="Warning threshold",
            annotation_position="top left",
            annotation_font=dict(color="#f59e0b"),
        )
    if forecast["critical_threshold"] is not None:
        fig.add_hline(
            y=forecast["critical_threshold"],
            line=dict(color="#ef4444", dash="dot"),
            annotation_text="Critical threshold",
            annotation_position="bottom left",
            annotation_font=dict(color="#ef4444"),
        )
    fig.add_annotation(
        x=forecast["forecast_times"][-1],
        y=forecast["predicted"][-1],
        text=f"{forecast['risk_level']} • {forecast['confidence_pct']}% confidence",
        showarrow=False,
        bgcolor="rgba(15,23,42,0.85)",
        bordercolor="#475569",
        borderwidth=1,
        font=dict(color="#e2e8f0", size=12),
        xanchor="right",
        yanchor="bottom",
    )
    fig.update_layout(
        title=f"{metric_name} Condition Forecast",
        xaxis_title="Time",
        yaxis_title=f"{metric_name} {units}",
        showlegend=True,
    )
    style_plotly_chart(fig)
    return fig


def render_dashboard():
    st.set_page_config(page_title="AI Robot Monitoring Demo", layout="wide")
    st.markdown(
        """
    <style>
      .app-header{display:flex;justify-content:space-between;align-items:center;padding:12px 0;margin-bottom:18px}
      .app-title{font-size:24px;font-weight:700;color:#f8fafc;margin:0}
      .app-sub{color:#94a3b8;font-size:13px;margin-top:4px}
      .app-status{color:#94a3b8;font-size:13px}
      .app-status .live-dot{height:10px;width:10px;background:#34d399;display:inline-block;border-radius:50%;margin-right:8px;vertical-align:middle}
      .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:16px}
      .kpi-card{background:#0f172a;border:1px solid rgba(148,163,184,0.14);border-radius:14px;padding:18px;box-shadow:0 12px 30px rgba(15,23,42,0.35);min-height:120px;transition:transform 0.2s ease}
      .kpi-card:hover{transform:translateY(-3px)}
      .kpi-label{color:#94a3b8;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px}
      .kpi-value{font-size:26px;font-weight:700;color:#f8fafc}
      .kpi-delta{color:#34d399;font-size:12px;margin-top:6px}
      .exec-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:20px}
      .exec-card{background:linear-gradient(135deg,#111827,#1f2937);border:1px solid rgba(148,163,184,0.18);border-radius:18px;padding:20px;box-shadow:0 16px 34px rgba(15,23,42,0.35);min-height:140px;position:relative;overflow:hidden}
      .exec-label{color:#94a3b8;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;display:block}
      .exec-value{font-size:32px;font-weight:800;color:#f8fafc;line-height:1.1}
      .exec-note{color:#cbd5e1;font-size:12px;margin-top:10px}
      .exec-pill{display:inline-flex;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700}
      .pill-green{background:#134e4a;color:#bef264}
      .pill-yellow{background:#713f12;color:#fde68a}
      .pill-red{background:#7f1d1d;color:#fecaca}
      .pill-blue{background:#1e3a8a;color:#bfdbfe}
      .robot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:18px;margin-bottom:20px}
      .robot-card{background:#111827;border:1px solid rgba(148,163,184,0.12);border-radius:16px;padding:18px;min-height:160px;position:relative;overflow:hidden;transition:transform 0.2s ease}
      .robot-card:hover{transform:translateY(-4px)}
      .robot-title{font-size:14px;font-weight:700;color:#f8fafc;margin-bottom:8px}
      .robot-score{font-size:28px;font-weight:700;color:#f8fafc;margin-bottom:6px}
      .robot-status{font-size:12px;text-transform:uppercase;letter-spacing:0.08em;padding:6px 10px;border-radius:999px;display:inline-flex;align-items:center}
      .status-Healthy{background:#134e4a;color:#a7f3d0}
      .status-Warning{background:#78350f;color:#ffedd5}
      .status-Critical{background:#7f1d1d;color:#fecaca}
      .robot-meta{color:#94a3b8;font-size:12px;line-height:1.5;margin-top:10px}
      .section-title{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin:24px 0 12px}
      .section-title h2{margin:0;color:#f8fafc;font-size:20px}
      .section-title span{color:#94a3b8;font-size:13px}
      .detail-summary{color:#cbd5e1;font-size:13px;margin-bottom:12px}
      .streamlit-expanderHeader{color:#f8fafc}
      .stDataFrame table{background:#0f172a;color:#e2e8f0}
    </style>
    """,
        unsafe_allow_html=True,
    )
    now_str = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(
        f"""
    <div class='app-header'>
      <div>
        <div class='app-title'>🤖 AI Robot Monitoring</div>
        <div class='app-sub'>Predictive Maintenance Dashboard</div>
      </div>
      <div class='app-status'>
        <span class='live-dot'></span>Live • Last Updated {now_str}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Controls
    with st.sidebar:
        st.header("Controls")
        interval = st.slider("Update interval (seconds)", 1, 10, st.session_state.interval)
        prediction_horizon = st.slider("Prediction horizon (seconds)", 10, 300, st.session_state.prediction_horizon)
        run_toggle = st.button("Start Live" if not st.session_state.running else "Pause Live")
        if run_toggle:
            st.session_state.running = not st.session_state.running
        st.markdown("---")
        selected_robot_id = st.selectbox(
            "Selected Robot",
            [r["id"] for r in st.session_state.robots],
            index=[r["id"] for r in st.session_state.robots].index(st.session_state.get("selected_robot_id", "Robot-01")) if st.session_state.get("selected_robot_id") else 0,
        )
        st.markdown("---")
        st.metric("Live status", "On" if st.session_state.running else "Off")
        st.metric("Update interval", f"{interval}s")
        st.markdown("---")
        st.write("Simulation settings")
        st.write(f"Robots: {NUM_ROBOTS}")

    st.session_state.interval = interval
    st.session_state.prediction_horizon = prediction_horizon
    st.session_state.selected_robot_id = selected_robot_id

    # Top KPI cards
    total = len(st.session_state.robots)
    healthy = sum(1 for r in st.session_state.robots if r["status"] == "Healthy")
    warning = sum(1 for r in st.session_state.robots if r["status"] == "Warning")
    critical = sum(1 for r in st.session_state.robots if r["status"] == "Critical")
    availability = st.session_state.avail_history[-1] if st.session_state.avail_history else 100.0
    downtime_pct = (st.session_state.downtime_hours / max(1.0, sum(r["runtime_hours"] for r in st.session_state.robots))) * 100.0
    alarm_count = st.session_state.alarm_count
    mtbf = (sum(r["runtime_hours"] for r in st.session_state.robots) / max(1, alarm_count)) if alarm_count > 0 else sum(r["runtime_hours"] for r in st.session_state.robots)

    st.markdown(
        f"""
    <div class='section-title'>
      <h2>Cluster Performance</h2>
      <span>Aggregated robot health metrics</span>
    </div>
    <div class='kpi-grid'>
      <div class='kpi-card'><div class='kpi-label'>Total Robots</div><div class='kpi-value'>{total}</div><div class='kpi-delta'>All Robots</div></div>
      <div class='kpi-card'><div class='kpi-label'>Healthy</div><div class='kpi-value'>{healthy}</div><div class='kpi-delta'>{healthy/total*100:.0f}%</div></div>
      <div class='kpi-card'><div class='kpi-label'>Warning</div><div class='kpi-value'>{warning}</div><div class='kpi-delta'>{warning/total*100:.0f}%</div></div>
      <div class='kpi-card'><div class='kpi-label'>Critical</div><div class='kpi-value'>{critical}</div><div class='kpi-delta'>{critical/total*100:.0f}%</div></div>
    </div>
    <div class='kpi-grid'>
      <div class='kpi-card'><div class='kpi-label'>Availability %</div><div class='kpi-value'>{availability:.1f}%</div><div class='kpi-delta'>Stable</div></div>
      <div class='kpi-card'><div class='kpi-label'>MTBF (hrs)</div><div class='kpi-value'>{mtbf:.1f}</div><div class='kpi-delta'>Estimate</div></div>
      <div class='kpi-card'><div class='kpi-label'>Downtime %</div><div class='kpi-value'>{downtime_pct:.2f}%</div><div class='kpi-delta'>Recent trend</div></div>
      <div class='kpi-card'><div class='kpi-label'>Alarm Count</div><div class='kpi-value'>{alarm_count}</div><div class='kpi-delta'>Alerts</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    pred_alerts = predictive_assessment(horizon_seconds=prediction_horizon)
    active_alerts = [r for r in st.session_state.robots if r["status"] != "Healthy"]
    active_alert_count = len(active_alerts)
    predicted_failures_count = len(pred_alerts)
    prev_availability = st.session_state.avail_history[-2] if len(st.session_state.avail_history) >= 2 else None
    prev_downtime_pct = st.session_state.downtime_pct_history[-2] if len(st.session_state.downtime_pct_history) >= 2 else None
    prev_pred_failures = st.session_state.pred_failures_history[-2] if len(st.session_state.pred_failures_history) >= 2 else None
    prev_active_alerts = st.session_state.active_alerts_history[-2] if len(st.session_state.active_alerts_history) >= 2 else None
    avail_arrow = trend_arrow(availability, prev_availability, positive_good=True)
    downtime_arrow = trend_arrow(downtime_pct, prev_downtime_pct, positive_good=False)
    pred_failures_arrow = trend_arrow(predicted_failures_count, prev_pred_failures, positive_good=False)
    active_alerts_arrow = trend_arrow(active_alert_count, prev_active_alerts, positive_good=False)
    st.session_state.downtime_pct_history.append(downtime_pct)
    st.session_state.pred_failures_history.append(predicted_failures_count)
    st.session_state.active_alerts_history.append(active_alert_count)

    st.markdown(
        f"""
    <div class='section-title'>
      <h2>Executive Summary</h2>
      <span>High-level operational KPIs for leadership review</span>
    </div>
    <div class='exec-grid'>
      <div class='exec-card'><span class='exec-label'>Total Robots</span><div class='exec-value'>{total}</div><span class='exec-note'>Entire monitored fleet</span></div>
      <div class='exec-card'><span class='exec-label'>Healthy Robots</span><div class='exec-value'>{healthy}</div><span class='exec-note'><span class='exec-pill pill-green'>Healthy {healthy/total*100:.0f}%</span></span></div>
      <div class='exec-card'><span class='exec-label'>Warning Robots</span><div class='exec-value'>{warning}</div><span class='exec-note'><span class='exec-pill pill-yellow'>Review required</span></span></div>
      <div class='exec-card'><span class='exec-label'>Critical Robots</span><div class='exec-value'>{critical}</div><span class='exec-note'><span class='exec-pill pill-red'>Immediate action</span></span></div>
      <div class='exec-card'><span class='exec-label'>Availability %</span><div class='exec-value'>{availability:.1f}%</div><span class='exec-note'>{avail_arrow} uptime trend</span></div>
      <div class='exec-card'><span class='exec-label'>MTBF</span><div class='exec-value'>{mtbf:.1f}</div><span class='exec-note'>{trend_arrow(mtbf, None, positive_good=True)} mean time between failures</span></div>
      <div class='exec-card'><span class='exec-label'>Downtime %</span><div class='exec-value'>{downtime_pct:.2f}%</div><span class='exec-note'>{downtime_arrow} operational downtime</span></div>
      <div class='exec-card'><span class='exec-label'>Predicted Failures</span><div class='exec-value'>{predicted_failures_count}</div><span class='exec-note'>{pred_failures_arrow} risk forecast</span></div>
      <div class='exec-card'><span class='exec-label'>Active Alerts</span><div class='exec-value'>{active_alert_count}</div><span class='exec-note'>{active_alerts_arrow} live issues</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    avg_health = float(np.mean([r["health_score"] for r in st.session_state.robots]))
    gauge_cols = st.columns([1, 1, 1])
    gauge_cols[0].plotly_chart(build_gauge("Average Fleet Health", avg_health, min_val=0, max_val=100, threshold=80, suffix=" pts", subtext="Aggregate robot wellness rating"), use_container_width=True)
    gauge_cols[1].plotly_chart(build_gauge("Critical Load", critical / total * 100 if total else 0, min_val=0, max_val=100, threshold=20, suffix="%", subtext="% of robots in critical state"), use_container_width=True)
    gauge_cols[2].plotly_chart(build_gauge("Downtime Risk", downtime_pct, min_val=0, max_val=20, threshold=5, suffix="%", subtext="Recent downtime exposure"), use_container_width=True)

    st.markdown("### Fleet Risk Overview")
    risk_cols = st.columns([1, 1.3])
    risk_cols[0].plotly_chart(build_alarm_pie(st.session_state.robots), use_container_width=True)
    risk_cols[1].plotly_chart(build_risk_ranking_chart(st.session_state.robots), use_container_width=True)

    st.markdown("### Robot Health Bar Chart")
    st.plotly_chart(build_robot_health_bar(st.session_state.robots), use_container_width=True)

    selected_robot = next(
        (r for r in st.session_state.robots if r["id"] == selected_robot_id),
        st.session_state.robots[0],
    )
    selected_predictions = predict_robot_metrics(selected_robot, horizon_seconds=st.session_state.prediction_horizon)

    st.markdown(
        """
    <div class='section-title'>
      <h2>Selected Robot Detail</h2>
      <span>Current readings and risk forecast</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    detail_cols = st.columns([1, 1, 1, 1])
    detail_cols[0].metric("Robot", selected_robot["id"])
    detail_cols[1].metric("Status", selected_robot["status"])
    detail_cols[2].metric("Health Score", selected_robot["health_score"])
    detail_cols[3].metric("Prediction Horizon", f"{st.session_state.prediction_horizon}s")
    st.markdown(
        f"**Current readings:** Temp {selected_robot['temperature']:.1f}°C • Torque {selected_robot['torque']:.1f} • Vib {selected_robot['vibration']:.1f} • Speed {selected_robot['speed']:.0f}"
    )
    st.markdown(
        f"**Risk score:** {selected_robot['risk_score']} • **Failure probability:** {selected_robot['predicted_failure_probability']}%"
    )

    if selected_robot["status"] != "Healthy":
        st.warning(f"Active AI rule triggers: {selected_robot['alert_summary']}")

    if selected_predictions:
        sp_cols = st.columns(3)
        for i, metric in enumerate(["temperature", "torque", "vibration"]):
            pred = selected_predictions[metric]
            sp_cols[i].metric(
                f"Predicted {metric.title()}",
                f"{pred['predicted']:.1f}",
                delta=f"Threshold {pred['threshold']}",
                help=f"Expected {metric} in {st.session_state.prediction_horizon}s: {pred['level']}",
            )

        st.markdown("### AI Condition Forecast")
        forecast_cols = st.columns(3)
        forecast_specs = [
            ("temperature", "Temperature", "C", "#f97316", 80, None),
            ("torque", "Torque", "Nm", "#60a5fa", 85, None),
            ("vibration", "Vibration", "mm/s", "#38bdf8", 70, None),
        ]
        for i, (metric, title, units, color, warning_thr, critical_thr) in enumerate(forecast_specs):
            forecast = forecast_robot_metric(
                selected_robot,
                metric,
                horizon_seconds=st.session_state.prediction_horizon,
                warning_threshold=warning_thr,
                critical_threshold=critical_thr,
            )
            if forecast:
                forecast_cols[i].plotly_chart(build_future_prediction_chart(forecast, title, units, color), use_container_width=True)
                forecast_cols[i].markdown(
                    f"**Predicted failure:** {forecast['predicted_failure_time']}  \n"
                    f"**Risk level:** {forecast['risk_level']}  \n"
                    f"**Confidence:** {forecast['confidence_pct']}%"
                )
            else:
                forecast_cols[i].info("Not enough history to generate a forecast yet.")
    else:
        st.info("Not enough history to predict the selected robot yet.")

    robot_cards = []
    sorted_robots = sorted(
        st.session_state.robots,
        key=lambda r: (0 if r["status"] == "Critical" else 1 if r["status"] == "Warning" else 2, -r["temperature"]),
    )[:8]
    for r in sorted_robots:
        score = r["health_score"]
        robot_cards.append(
            f"""
      <div class='robot-card'>
        <div class='robot-title'>{r['id']}</div>
        <div class='robot-score'>{score}</div>
        <div class='robot-status status-{r['status']}'>{r['status']}</div>
        <div class='robot-meta'>Risk: {r['risk_score']} &bull; Failure: {r['predicted_failure_probability']}%<br>{r['alert_summary']}</div>
        <div class='robot-meta'>Temp: {r['temperature']:.1f}°C • Torque: {r['torque']:.1f} • Vib: {r['vibration']:.1f}</div>
      </div>
      """
        )

    st.markdown(
        """
    <div style='display:flex;align-items:center;justify-content:space-between;margin:22px 0 14px'>
      <div style='color:#f8fafc;font-size:18px;font-weight:700'>Robot Health Overview</div>
      <div style='color:#94a3b8;font-size:13px'>Showing top 8 robots by risk</div>
    </div>
    <div class='robot-grid'>
    """ + "".join(robot_cards) + """
    </div>
    """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    k1.metric("Total Robots", total)
    k2.metric("Healthy", healthy, delta=f"{healthy/total*100:.1f}%")
    k3.metric("Warning", warning, delta=f"{warning/total*100:.1f}%")
    k4.metric("Critical", critical, delta=f"{critical/total*100:.1f}%")

    # Business KPIs row
    b1, b2, b3, b4 = st.columns(4)
    last_avail = st.session_state.avail_history[-1] if st.session_state.avail_history else 100.0
    last_health = st.session_state.health_history[-1] if st.session_state.health_history else 100.0
    b1.metric("Availability %", f"{last_avail:.1f}%")
    mtbf = (sum(r["runtime_hours"] for r in st.session_state.robots) / max(1, st.session_state.alarm_count)) if st.session_state.alarm_count > 0 else sum(r["runtime_hours"] for r in st.session_state.robots)
    b2.metric("MTBF (hrs)", f"{mtbf:.1f}")
    downtime_pct = (st.session_state.downtime_hours / max(1.0, sum(r["runtime_hours"] for r in st.session_state.robots))) * 100.0
    b3.metric("Downtime %", f"{downtime_pct:.2f}%")
    b4.metric("Alarm Count", st.session_state.alarm_count)

    # Live KPI cards
    st.markdown("### Live KPIs")
    l1, l2, l3, l4 = st.columns(4)
    # compute current aggregated averages
    avg_temp = np.mean([r["temperature"] for r in st.session_state.robots])
    avg_torque = np.mean([r["torque"] for r in st.session_state.robots])
    avg_speed = np.mean([r["speed"] for r in st.session_state.robots])
    avg_vib = np.mean([r["vibration"] for r in st.session_state.robots])
    l1.metric("Avg Temperature (°C)", f"{avg_temp:.1f}")
    l2.metric("Avg Torque", f"{avg_torque:.1f}")
    l3.metric("Avg Speed", f"{avg_speed:.0f}")
    l4.metric("Avg Vibration", f"{avg_vib:.1f}")

    # Trend charts
    st.markdown(
        """
    <div class='section-title'>
      <h2>Trends</h2>
      <span>Cluster availability and metric trajectories</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    ts = list(st.session_state.ts_history)
    downtime_history = list(st.session_state.downtime_pct_history)
    avail = list(st.session_state.avail_history)
    health = list(st.session_state.health_history)

    trend_col_1, trend_col_2 = st.columns(2)
    with trend_col_1:
        if ts and avail:
            fig_avail = build_metric_trend_chart(ts, avail, "Availability", "Availability %", "#22c55e")
            fig_avail.update_layout(title_text="Availability % Over Time")
            st.plotly_chart(fig_avail, use_container_width=True)
        else:
            st.info("Availability history not yet available")

        if ts and health:
            fig_health = build_metric_trend_chart(ts, health, "Health Score", "Health Score", "#38bdf8")
            fig_health.update_layout(title_text="Health Score Over Time")
            st.plotly_chart(fig_health, use_container_width=True)
        else:
            st.info("Health history not yet available")

    with trend_col_2:
        if ts and downtime_history:
            fig_downtime = build_downtime_trend(ts, downtime_history)
            st.plotly_chart(fig_downtime, use_container_width=True)
        else:
            st.info("Downtime trend history not yet available")

        if len(st.session_state.ts_history) > 0:
            L = len(st.session_state.ts_history)
            temps_series = [np.mean([r["history"]["temperature"][i] for r in st.session_state.robots]) for i in range(L)]
            torque_series = [np.mean([r["history"]["torque"][i] for r in st.session_state.robots]) for i in range(L)]
            vib_series = [np.mean([r["history"]["vibration"][i] for r in st.session_state.robots]) for i in range(L)]
            fig_metrics = px.line(
                pd.DataFrame({"ts": list(st.session_state.ts_history), "Avg Temp": temps_series, "Avg Torque": torque_series, "Avg Vib": vib_series}),
                x="ts",
                y=["Avg Temp", "Avg Torque", "Avg Vib"],
                labels={"ts": "Time"},
                title="Avg Temp / Torque / Vibration Over Time",
            )
            fig_metrics.update_traces(mode="lines+markers", hovertemplate="%{x}<br>%{y:.2f}")
            style_plotly_chart(fig_metrics)
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            curr_avg_temp = np.mean([r["temperature"] for r in st.session_state.robots])
            curr_avg_torque = np.mean([r["torque"] for r in st.session_state.robots])
            curr_avg_vib = np.mean([r["vibration"] for r in st.session_state.robots])
            df_now = pd.DataFrame({"metric": ["Avg Temp", "Avg Torque", "Avg Vib"], "value": [curr_avg_temp, curr_avg_torque, curr_avg_vib]})
            fig_now = px.bar(df_now, x="metric", y="value", title="Current Aggregated Metrics")
            style_plotly_chart(fig_now)
            st.plotly_chart(fig_now, use_container_width=True)

    metric_cols = st.columns(3)
    if ts:
        metric_cols[0].plotly_chart(build_metric_trend_chart(ts, [np.mean([r["history"]["temperature"][i] for r in st.session_state.robots]) for i in range(len(ts))], "Temperature", "Temperature °C", "#f97316"), use_container_width=True)
        metric_cols[1].plotly_chart(build_metric_trend_chart(ts, [np.mean([r["history"]["torque"][i] for r in st.session_state.robots]) for i in range(len(ts))], "Torque", "Torque", "#38bdf8"), use_container_width=True)
        metric_cols[2].plotly_chart(build_metric_trend_chart(ts, [np.mean([r["history"]["vibration"][i] for r in st.session_state.robots]) for i in range(len(ts))], "Vibration", "Vibration", "#60a5fa"), use_container_width=True)
    else:
        metric_cols[0].info("Temperature history loading...")
        metric_cols[1].info("Torque history loading...")
        metric_cols[2].info("Vibration history loading...")

    # Active alerts panel
    st.markdown(
        """
    <div class='section-title'>
      <h2>Active Alerts</h2>
      <span>Real-time issues requiring attention</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    alerts = [r for r in st.session_state.robots if r["status"] != "Healthy"]
    if alerts:
        for r in sorted(alerts, key=lambda x: (x['status'] != 'Critical', -x['temperature'])):
            st.caption(f"AI triggers: {r['alert_summary']} | Health {r['health_score']} | Risk {r['risk_score']} | Failure {r['predicted_failure_probability']}%")
            if r['status'] == 'Critical':
                st.error(
                    f"{r['id']} — {r['status']}: Temp={r['temperature']:.1f}°C, Torque={r['torque']:.1f}, Vib={r['vibration']:.1f} • Risk {r['risk_score']} • Failure {r['predicted_failure_probability']}%"
                )
            else:
                st.warning(
                    f"{r['id']} — {r['status']}: Temp={r['temperature']:.1f}°C, Torque={r['torque']:.1f}, Vib={r['vibration']:.1f} • Risk {r['risk_score']} • Failure {r['predicted_failure_probability']}%"
                )
    else:
        st.success("No active alerts — all robots healthy")

    # Predictive alerts
    st.markdown(
        """
    <div class='section-title'>
      <h2>Predictive Alerts</h2>
      <span>Forecasted threshold breaches for the next period</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    pred_alerts = predictive_assessment(horizon_seconds=prediction_horizon)
    if pred_alerts:
        # show criticals first
        crits = [a for a in pred_alerts if a['level'] == 'Critical']
        warns = [a for a in pred_alerts if a['level'] == 'Warning']
        if crits:
            for a in crits:
                st.error(f"PREDICTED CRITICAL: {a['id']} — {a['metric']} -> {a['predicted']:.1f} (threshold {a['threshold']}) in {prediction_horizon}s")
        if warns:
            for a in warns:
                st.warning(f"Predicted: {a['id']} — {a['metric']} -> {a['predicted']:.1f} (threshold {a['threshold']}) in {prediction_horizon}s")
    else:
        st.info("No predicted threshold breaches in the selected horizon")

    # Alerts summary table
    st.markdown(
        """
    <div class='section-title'>
      <h2>Alerts Summary</h2>
      <span>Full fleet alert and risk matrix</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    status_order = {"Critical": 0, "Warning": 1, "Healthy": 2}
    prediction_map = {}
    for a in pred_alerts:
        prediction_map.setdefault(a["id"], []).append(f"{a['metric']} ({a['level']})")

    alert_rows = []
    for r in st.session_state.robots:
        alert_rows.append(
            {
                "Robot": r["id"],
                "Status": r["status"],
                "Temp (°C)": f"{r['temperature']:.1f}",
                "Torque": f"{r['torque']:.1f}",
                "Vib": f"{r['vibration']:.1f}",
                "Health": r["health_score"],
                "Risk Score": r["risk_score"],
                "Failure Prob %": f"{r['predicted_failure_probability']}%",
                "AI Rule Triggers": r["alert_summary"],
                "Predicted Risk": ", ".join(prediction_map.get(r["id"], [])) or "None",
                "StatusOrder": status_order.get(r["status"], 3),
            }
        )
    df_alerts = pd.DataFrame(alert_rows)
    df_alerts = df_alerts.sort_values(["StatusOrder", "Robot"]).drop(columns=["StatusOrder"])
    st.dataframe(df_alerts, height=320)

    # Correlation heatmap
    st.markdown(
        """
    <div class='section-title'>
      <h2>Parameter Correlation</h2>
      <span>Relationships across key sensor metrics</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    metric_df = pd.DataFrame(
        {
            "temperature": [r["temperature"] for r in st.session_state.robots],
            "torque": [r["torque"] for r in st.session_state.robots],
            "vibration": [r["vibration"] for r in st.session_state.robots],
            "speed": [r["speed"] for r in st.session_state.robots],
        }
    )
    corr = metric_df.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Live Parameter Correlation",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def main_loop():
    init_session()
    render_dashboard()

    # live loop runs while toggle is on
    while st.session_state.running:
        results = update_robots(interval_seconds=st.session_state.interval)
        # short sleep to simulate live updates
        time.sleep(st.session_state.interval)
        # force a rerun to refresh UI; use safe fallback if API changed
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException

            raise RerunException()
        except Exception:
            # final fallback: stop this run — UI will refresh on next interaction
            st.stop()


if __name__ == "__main__":
    main_loop()
