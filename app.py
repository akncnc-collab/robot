import time
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Simple AI-based Robot Monitoring Demo (single-file)
# Minimal deps: streamlit, plotly, numpy


NUM_ROBOTS = 20
HISTORY_LENGTH = 60


def init_session():
    if "robots" not in st.session_state:
        robots = []
        for i in range(1, NUM_ROBOTS + 1):
            base_temp = random.uniform(55, 75)
            base_torque = random.uniform(45, 70)
            base_speed = random.uniform(900, 1400)
            base_vib = random.uniform(15, 40)
            runtime_hours = random.uniform(100, 2000)
            r = {
                "id": f"Robot-{i:02d}",
                "temperature": base_temp,
                "torque": base_torque,
                "speed": base_speed,
                "vibration": base_vib,
                "runtime_hours": runtime_hours,
                "status": "Healthy",
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
            robots.append(r)
        st.session_state.robots = robots
        st.session_state.running = False
        st.session_state.alarm_count = 0
        st.session_state.downtime_hours = 0.0
        st.session_state.avail_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.health_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.ts_history = deque(maxlen=HISTORY_LENGTH)
        st.session_state.interval = 2
        st.session_state.prediction_horizon = 60


def ai_health_check(robot):
    # Business AI rules (simple thresholds)
    temp = robot["temperature"]
    vib = robot["vibration"]
    torque = robot["torque"]
    status = "Healthy"
    if torque > 85:
        status = "Critical"
    elif temp > 80 or vib > 70:
        status = "Warning"
    return status


def robot_health_score(robot):
    # Estimate a health score from metrics for dashboard display
    temp = robot["temperature"]
    torque = robot["torque"]
    vib = robot["vibration"]
    score = 100.0
    score -= max(0.0, (temp - 60.0) * 0.7)
    score -= max(0.0, (torque - 55.0) * 0.6)
    score -= max(0.0, (vib - 30.0) * 0.8)
    score = np.clip(score, 20.0, 100.0)
    return int(score)


def update_robots(interval_seconds=2.0):
    now = datetime.now()
    avg_temp = avg_torque = avg_speed = avg_vib = 0.0
    critical_count = warning_count = 0
    for r in st.session_state.robots:
        # Small drift plus occasional spikes
        r["temperature"] += random.uniform(-0.5, 0.8)
        if random.random() < 0.02:
            r["temperature"] += random.uniform(5, 15)  # spike
        r["torque"] += random.uniform(-0.7, 0.7)
        if random.random() < 0.015:
            r["torque"] += random.uniform(10, 25)
        r["speed"] += random.uniform(-10, 10)
        r["vibration"] += random.uniform(-0.6, 0.6)
        if random.random() < 0.02:
            r["vibration"] += random.uniform(5, 20)

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
        prev_status = r["status"]
        new_status = ai_health_check(r)
        r["status"] = new_status
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
    # Health score: scaled 0-100 (Healthy=100, Warning=60, Critical=20) average
    score_map = {"Healthy": 100, "Warning": 60, "Critical": 20}
    health_scores = [score_map[r["status"]] for r in st.session_state.robots]
    health_score = float(np.mean(health_scores))

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
        for metric, thresh, level_when_exceed in [("temperature", 80, "Warning"), ("vibration", 70, "Warning"), ("torque", 85, "Critical")]:
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

            # check thresholds
            if metric == "torque":
                if predicted > thresh:
                    alerts.append({"id": r["id"], "metric": metric, "predicted": predicted, "threshold": thresh, "level": "Critical"})
            else:
                if predicted > thresh:
                    alerts.append({"id": r["id"], "metric": metric, "predicted": predicted, "threshold": thresh, "level": "Warning"})
    return alerts


def predict_robot_metrics(robot, horizon_seconds=60):
    ts_list = list(robot["history"]["ts"])
    if len(ts_list) < 2:
        return {}
    t0 = ts_list[0]
    secs = np.array([(t - t0).total_seconds() for t in ts_list])
    predictions = {}
    for metric, thresh, critical_if in [("temperature", 80, "Warning"), ("vibration", 70, "Warning"), ("torque", 85, "Critical")]:
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
        level = "Healthy"
        if metric == "torque" and predicted > thresh:
            level = "Critical"
        elif metric != "torque" and predicted > thresh:
            level = "Warning"
        predictions[metric] = {
            "predicted": predicted,
            "threshold": thresh,
            "level": level,
        }
    return predictions


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
    detail_cols[2].metric("Health Score", robot_health_score(selected_robot))
    detail_cols[3].metric("Prediction Horizon", f"{st.session_state.prediction_horizon}s")
    st.markdown(
        f"**Current readings:** Temp {selected_robot['temperature']:.1f}°C • Torque {selected_robot['torque']:.1f} • Vib {selected_robot['vibration']:.1f} • Speed {selected_robot['speed']:.0f}"
    )

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
    else:
        st.info("Not enough history to predict the selected robot yet.")

    robot_cards = []
    sorted_robots = sorted(
        st.session_state.robots,
        key=lambda r: (0 if r["status"] == "Critical" else 1 if r["status"] == "Warning" else 2, -r["temperature"]),
    )[:8]
    for r in sorted_robots:
        score = robot_health_score(r)
        robot_cards.append(
            f"""
      <div class='robot-card'>
        <div class='robot-title'>{r['id']}</div>
        <div class='robot-score'>{score}</div>
        <div class='robot-status status-{r['status']}'>{r['status']}</div>
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
    col1, col2 = st.columns(2)
    ts = list(st.session_state.ts_history)
    if not ts:
        # create initial from robot histories
        ts = [r["history"]["ts"][-1] for r in st.session_state.robots][:HISTORY_LENGTH]

    with col1:
        ts = list(st.session_state.ts_history)
        avail = list(st.session_state.avail_history)
        health = list(st.session_state.health_history)
        # ensure matching lengths
        n_avail = min(len(ts), len(avail))
        if n_avail > 0:
            df_avail = pd.DataFrame({"ts": ts[-n_avail:], "availability": avail[-n_avail:]})
            fig_temp = px.line(df_avail, x="ts", y="availability", labels={"ts": "Time", "availability": "Availability %"}, title="Availability % Over Time")
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Availability history not yet available")

        n_health = min(len(ts), len(health))
        if n_health > 0:
            df_health = pd.DataFrame({"ts": ts[-n_health:], "health": health[-n_health:]})
            fig_health = px.line(df_health, x="ts", y="health", labels={"ts": "Time", "health": "Health Score"}, title="Health Score Over Time")
            st.plotly_chart(fig_health, use_container_width=True)
        else:
            st.info("Health history not yet available")

    with col2:
        # Aggregated metric trends: compute per-timestamp averages across robots
        if len(st.session_state.ts_history) > 0:
            L = len(st.session_state.ts_history)
            temps_series = [np.mean([r["history"]["temperature"][i] for r in st.session_state.robots]) for i in range(L)]
            torque_series = [np.mean([r["history"]["torque"][i] for r in st.session_state.robots]) for i in range(L)]
            vib_series = [np.mean([r["history"]["vibration"][i] for r in st.session_state.robots]) for i in range(L)]
            df_metrics = pd.DataFrame({"ts": list(st.session_state.ts_history), "avg_temp": temps_series, "avg_torque": torque_series, "avg_vib": vib_series})
            fig_metrics = px.line(df_metrics, x="ts", y=["avg_temp", "avg_torque", "avg_vib"], labels={"ts": "Time"}, title="Avg Temp / Torque / Vibration Over Time")
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            # show current snapshot as fallback
            curr_avg_temp = np.mean([r["temperature"] for r in st.session_state.robots])
            curr_avg_torque = np.mean([r["torque"] for r in st.session_state.robots])
            curr_avg_vib = np.mean([r["vibration"] for r in st.session_state.robots])
            df_now = pd.DataFrame({"metric": ["Avg Temp", "Avg Torque", "Avg Vib"], "value": [curr_avg_temp, curr_avg_torque, curr_avg_vib]})
            fig_now = px.bar(df_now, x="metric", y="value", title="Current Aggregated Metrics")
            st.plotly_chart(fig_now, use_container_width=True)

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
            st.warning(f"{r['id']} — {r['status']}: Temp={r['temperature']:.1f}°C, Torque={r['torque']:.1f}, Vib={r['vibration']:.1f}")
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
                "Health": robot_health_score(r),
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
