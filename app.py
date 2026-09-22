"""Mini NWDAF: a research-paper implementation for a classroom demonstration."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import transitions
from event_engine import ReplaySession, to_events
from predictor import Predictor
from preprocessing import ROOT, load_dataset

st.set_page_config(page_title="Mini NWDAF | Mobility Intelligence", page_icon="📡", layout="wide")
st.markdown("""<style>
.block-container {max-width:1280px; padding-top:2.3rem; padding-bottom:3rem;}
h1 {letter-spacing:-1.6px; font-weight:750 !important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:white;border:1px solid #DFE6EE;border-radius:12px;padding:18px 20px;}
[data-testid="stMetricLabel"] {color:#607086;font-size:13px;}
[data-testid="stMetricValue"] {font-size:29px;}
[data-testid="stSidebar"] {border-right:1px solid #DFE6EE;}
.eyebrow {color:#087F8C;font-size:12px;font-weight:750;letter-spacing:2px;margin-bottom:9px;}
.intro {color:#607086;font-size:17px;max-width:760px;margin-bottom:25px;}
.tag {display:inline-block;background:#E3F1EF;color:#086A65;border-radius:5px;padding:5px 10px;font-size:12px;font-weight:650;}
.footnote {color:#738197;font-size:12px; margin-top:20px;}
</style>""", unsafe_allow_html=True)


def short(cell):
    return str(cell).lstrip("0") or "0"


def device_name(supi):
    return "Device " + str(supi)[-2:]


def chart_style(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#30415D"),
                      margin=dict(l=10, r=15, t=20, b=15))
    return fig


@st.cache_data
def read_data(signature):
    return load_dataset()


@st.cache_resource
def read_predictor(signature):
    return Predictor()


def topology(current=None, predicted=None):
    # Illustrative topology, deliberately not used as model coordinates.
    positions = {"30": (0, 1), "60": (1, 1), "40": (0, 0), "50": (1, 0)}
    fig = go.Figure()
    for a, b in [("30", "60"), ("30", "40"), ("40", "50"), ("60", "50")]:
        fig.add_trace(go.Scatter(x=[positions[a][0], positions[b][0]], y=[positions[a][1], positions[b][1]],
                                 mode="lines", line=dict(color="#DCE5EE", width=2), hoverinfo="skip", showlegend=False))
    if current and predicted and short(current) in positions and short(predicted) in positions:
        a, b = positions[short(current)], positions[short(predicted)]
        fig.add_annotation(x=b[0], y=b[1], ax=a[0], ay=a[1], xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=4, arrowcolor="#E99B35", standoff=30, startstandoff=30)
    for cell, (x, y) in positions.items():
        color = "#087F8C" if current and short(current) == cell else "#E99B35" if predicted and short(predicted) == cell else "#C5D3E3"
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=53, color=color, line=dict(color="white", width=3)),
                                 text=[f"<b>{cell}</b>"], textposition="middle center", textfont=dict(color="white", size=16),
                                 hovertemplate=f"Cell {cell}<extra></extra>", showlegend=False))
    fig.update_layout(height=290, xaxis=dict(visible=False, range=[-.35, 1.35]),
                      yaxis=dict(visible=False, range=[-.35, 1.35]), dragmode=False)
    return chart_style(fig)


with st.sidebar:
    st.markdown("### 📡 Mini NWDAF")
    st.caption("5G MOBILITY INTELLIGENCE")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Event replay", "Model lab", "About the paper"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Research dataset**")
    st.caption("Authors' Free5GC / UERANSIM testbed\n\n03–18 February 2025")
    st.markdown('<span class="tag">LOCAL · EDUCATIONAL PROTOTYPE</span>', unsafe_allow_html=True)
    st.caption("UE = device · gNB = base station\n\nA handover moves a connection to another cell.")

if not all((ROOT / "data" / "raw" / name).exists() for name in ["df_location.csv", "df_reg.csv"]):
    st.title("Set up your research dataset")
    st.info("Download the two original CSV files from the authors' repository. No synthetic data is generated.")
    if st.button("Download authors' dataset", type="primary"):
        try:
            from download_data import download
            with st.spinner("Downloading the pinned source files…"):
                download()
            st.rerun()
        except Exception as error:
            st.error(f"Download failed: {error}. You can also run: python download_data.py")
    st.stop()

signature = tuple((ROOT / "data" / "raw" / name).stat().st_mtime_ns for name in ["df_location.csv", "df_reg.csv"])
try:
    locations, visits, examples, stats = read_data(signature)
except (ValueError, OSError) as error:
    st.error(f"Dataset could not be prepared: {error}")
    st.stop()

metrics_path = ROOT / "results" / "metrics.json"
model_path = ROOT / "models" / "gradient_boosting.joblib"
if not metrics_path.exists() or not model_path.exists():
    st.title("Train the mobility models")
    st.write("The dataset is ready. Train the baseline, Decision Tree and Gradient Boosting to enable the dashboard.")
    if st.button("Train models", type="primary"):
        from train import train_project
        with st.spinner("Preparing temporal holdout and training models…"):
            train_project()
        st.rerun()
    st.stop()

report = json.loads(metrics_path.read_text(encoding="utf-8"))
predictor = read_predictor(model_path.stat().st_mtime_ns)
routes = transitions(visits)
cutoff = pd.Timestamp(report["cutoff"])

if page == "Overview":
    st.markdown('<div class="eyebrow">NETWORK ANALYTICS / RESEARCH DEMO</div>', unsafe_allow_html=True)
    st.title("Understand movement. Anticipate the next cell.")
    st.markdown('<div class="intro">Explore recorded 5G device mobility, follow handovers, and test next-cell predictions on the authors’ own dataset.</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for col, label, value in zip(columns, ["Location records", "Devices observed", "Serving cells", "Within-session transitions"],
                                 [stats["raw_location_rows"], stats["devices"], stats["cells"], len(routes)]):
        col.metric(label, f"{value:,}")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.subheader("Where devices move")
        st.caption("Observed transitions after filtering inactive periods and repeated same-cell reports.")
        counts = routes.route.value_counts().rename_axis("Route").reset_index(name="Transitions")
        fig = px.bar(counts, x="Transitions", y="Route", orientation="h", color_discrete_sequence=["#087F8C"])
        fig.update_layout(height=320, yaxis=dict(autorange="reversed"))
        st.plotly_chart(chart_style(fig), width="stretch")
    with right:
        st.subheader("Four-cell testbed")
        st.caption("Illustrative arrangement · numbers identify serving cells")
        st.plotly_chart(topology(), width="stretch", config={"displayModeBar": False})
    st.subheader("A device's movement history")
    selected = st.selectbox("Select device", sorted(visits.supi.unique()), format_func=device_name)
    device_visits = visits[visits.supi == selected].copy()
    device_visits["Cell"] = device_visits.current_cell.map(short)
    device_visits["Session"] = device_visits.segment.astype(str)
    fig = px.scatter(device_visits, x="time", y="Cell", color="Cell", hover_data=["Session"],
                     color_discrete_sequence=["#087F8C", "#4263A8", "#E99B35", "#8B6BAD"])
    fig.update_layout(height=230, xaxis_title="Recorded time (source timezone unspecified)", showlegend=False)
    st.plotly_chart(chart_style(fig), width="stretch")
    with st.expander("Inspect source records and preparation counts"):
        st.json(stats)
        st.dataframe(locations.head(100), hide_index=True, width="stretch")
        st.caption("Original identifiers remain strings. The file has no ready-made next-cell label; labels are derived within each device/session.")
    st.info("Next: open Event replay to subscribe to location reports and watch predictions resolve as later events arrive.")

elif page == "Event replay":
    st.markdown('<div class="eyebrow">EVENT SUBSCRIPTION / HELD-OUT RECORDS</div>', unsafe_allow_html=True)
    st.title("Watch the next handover unfold.")
    st.markdown('<div class="intro">Subscribe, advance the recorded stream, and compare each forecast with the next observed cell. Predictions never see future events.</div>', unsafe_allow_html=True)
    selected = st.selectbox("Replay device", ["All devices"] + sorted(visits.supi.unique()),
                            format_func=lambda x: x if x == "All devices" else device_name(x))
    replay_key = (selected, model_path.stat().st_mtime_ns, signature)
    if st.session_state.get("replay_key") != replay_key:
        subset = visits if selected == "All devices" else visits[visits.supi == selected]
        st.session_state.replay = ReplaySession(to_events(subset[subset.time >= cutoff]), predictor,
                                                to_events(subset[subset.time < cutoff]))
        st.session_state.replay_key = replay_key
    replay = st.session_state.replay
    controls = st.columns([1, 1, 1, 1, 1.2])
    if controls[0].button("Subscribe", type="primary", disabled=replay.subscription_id is not None, width="stretch"):
        replay.subscribe()
        st.rerun()
    if controls[1].button("Next event", disabled=replay.cursor >= len(replay.events), width="stretch"):
        replay.step()
    if controls[2].button("Advance 10", disabled=replay.cursor >= len(replay.events), width="stretch"):
        for _ in range(10):
            replay.step()
    if controls[3].button("Unsubscribe", disabled=replay.subscription_id is None, width="stretch"):
        replay.unsubscribe()
        st.rerun()
    if controls[4].button("Reset replay", width="stretch"):
        del st.session_state.replay_key
        st.rerun()
    if replay.subscription_id:
        st.caption(f"● SUBSCRIBED · {replay.subscription_id} · LOCATION_REPORT")
    else:
        st.caption("○ NOT SUBSCRIBED · Events still advance, but no notifications reach analytics.")
    st.progress(replay.cursor / max(len(replay.events), 1), text=f"{replay.cursor} / {len(replay.events)} recorded events replayed")
    analysis = replay.analytics
    resolved = pd.DataFrame(analysis.resolved)
    cols = st.columns(3)
    cols[0].metric("Notifications received", analysis.received)
    cols[1].metric("Forecasts resolved", len(resolved))
    cols[2].metric("Replay accuracy", f"{resolved.correct.mean():.1%}" if not resolved.empty else "—")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("Latest delivered observation")
        latest = analysis.latest
        if latest:
            st.write(f"**{device_name(latest['supi'])}** · {latest['time']}")
            pair = st.columns(2)
            pair[0].metric("Current cell", short(latest["cell"]))
            forecast = latest["forecast"]
            pair[1].metric("Predicted next cell", short(forecast["cell"]) if forecast else "Waiting")
            if forecast:
                st.caption(f"Previous cell: {short(latest['previous_cell'])} · Model score: {forecast['score']:.1%}")
                st.caption("Model scores are uncalibrated estimates, not a guarantee. The actual next cell is revealed only by a later event.")
            else:
                st.info("New session or missing history: observe another cell before forecasting.")
        else:
            st.info("Subscribe, then select Next event to begin.")
    with right:
        latest = analysis.latest
        prediction = latest["forecast"]["cell"] if latest and latest["forecast"] else None
        st.plotly_chart(topology(latest["cell"] if latest else None, prediction), width="stretch", config={"displayModeBar": False})
        st.caption("Teal = observed current cell · Amber = predicted next cell · Illustrative topology")
    if replay.cursor == len(replay.events):
        st.success("Replay complete. Pending forecasts have no later observation and are not scored.")
    tabs = st.tabs(["Resolved forecasts", "Notification log", "How this replay works"])
    with tabs[0]:
        if resolved.empty:
            st.caption("Forecast outcomes appear after the next observation for the same device and session.")
        else:
            st.dataframe(resolved.iloc[::-1], hide_index=True, width="stretch")
            st.download_button("Download replay results", resolved.to_csv(index=False), "replay_results.csv", "text/csv")
    with tabs[1]:
        st.dataframe(pd.DataFrame(replay.log).iloc[::-1], hide_index=True, width="stretch")
    with tabs[2]:
        st.write("The publisher replays only time, device, cell and session metadata. A matching subscription invokes the analytics handler. The handler updates past history, resolves an earlier forecast if possible, then issues a new forecast.")
        st.write("History is seeded only from events before the evaluation cutoff. Unsubscribing clears history and pending forecasts because missed events break continuity. Device selection resets the replay.")
        st.caption("Local Python callbacks simulate subscriptions. This is not a live 5G connection or a standards-compliant AMF/NWDAF API.")

elif page == "Model lab":
    st.markdown('<div class="eyebrow">SUPERVISED LEARNING / CHRONOLOGICAL EVALUATION</div>', unsafe_allow_html=True)
    st.title("Measure what the model learns.")
    st.markdown('<div class="intro">Compare two classifiers with a simple transition baseline. All scores below come from held-out, later observations.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Training examples", report["train_rows"])
    cols[1].metric("Test examples", report["test_rows"])
    cols[2].metric("Boundary examples purged", report["purged_rows"])
    scores = pd.DataFrame([{"Model": name, "Accuracy": item["accuracy"], "Macro-F1": item["macro_f1"]}
                           for name, item in report["models"].items()])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Model comparison")
        fig = px.bar(scores, x="Model", y="Accuracy", color="Model", text=scores.Accuracy.map(lambda x: f"{x:.1%}"),
                     color_discrete_sequence=["#A3B3C7", "#4263A8", "#087F8C"])
        fig.update_layout(height=320, showlegend=False, yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(chart_style(fig), width="stretch")
        st.dataframe(scores.style.format({"Accuracy": "{:.2%}", "Macro-F1": "{:.3f}"}), hide_index=True, width="stretch")
    with right:
        st.subheader("Where predictions go wrong")
        name = st.selectbox("Model", list(report["models"]), index=2)
        labels = [short(x) for x in report["labels"]]
        fig = px.imshow(report["models"][name]["confusion_matrix"], x=labels, y=labels, text_auto=True,
                        labels=dict(x="Predicted cell", y="Actual cell", color="Examples"), color_continuous_scale="Teal")
        fig.update_layout(height=340)
        st.plotly_chart(chart_style(fig), width="stretch")
    st.info("The paper reports 80.65% for Gradient Boosting. Our features, session filtering and chronological split differ, so these scores are not an exact replication. The demo model is fixed in advance, not chosen using test scores.")
    with st.expander("Features, split and reproducibility", expanded=True):
        st.write("**Inputs:** device ID, current cell, previous cell, time-of-day category, and cyclical hour encoding.")
        st.write("**Target:** next different observed cell within the same registration segment.")
        st.write(f"**Cutoff:** {cutoff}. Earlier examples train the model only if their next-cell label also occurs before the cutoff.")
        st.write("**Gradient Boosting:** 100 estimators, maximum depth 9, learning rate 0.05, random seed 42. These hyperparameters follow Section IV-B; additional paper features are omitted.")
        st.write("This small dataset contains the same devices in training and testing. It evaluates later mobility of known devices, not unseen users. No hyperparameters are tuned against the held-out test set.")
    st.download_button("Download evaluation report", metrics_path.read_text(), "metrics.json", "application/json")
    if st.button("Retrain reproducibly"):
        from train import train_project
        with st.spinner("Training and evaluating…"):
            train_project()
        st.cache_resource.clear()
        st.session_state.pop("replay_key", None)
        st.rerun()

else:
    st.markdown('<div class="eyebrow">RESEARCH CONTEXT / IMPLEMENTATION BOUNDARIES</div>', unsafe_allow_html=True)
    st.title("A small implementation of a bigger idea.")
    st.write("**Enhanced Open-Source NWDAF for Event-Driven Analytics in 5G Networks**")
    st.write("Henok Daniel, Omar Alhussein, Jie Liang, Cheng Li, and Ernesto Damiani.")
    st.markdown("[Read the paper](https://arxiv.org/abs/2601.01838) · [Authors' repository](https://github.com/HenokDanielbfg/5g-testbed-conference)")
    st.table(pd.DataFrame([
        ["Event subscription / notification (III-B)", "Python publisher and subscribed analytics callback"],
        ["UE mobility analytics (IV-A)", "Routes and device timelines from the authors' recorded events"],
        ["Next-cell classification (IV-B)", "Decision Tree, Gradient Boosting, and transition baseline"],
        ["Full Free5GC / UERANSIM network", "Not deployed; recorded testbed data is replayed"],
        ["Additional features / SMF session services", "Outside this small implementation"]], columns=["Paper concept", "This project"]))
    st.subheader("The implementation in one flow")
    st.code("Authors' CSVs → clean + session boundaries → past-only features\n                           ↓                       ↓\n                 recorded event replay       train → evaluate\n                           ↓                       ↓\n                 subscription callback ← saved ML model\n                           ↓\n                 next-cell forecast + dashboard", language=None)
    st.subheader("A two-minute classroom demonstration")
    st.write("1. Show the original data and route counts in Overview.\n2. Open Model lab and explain the chronological test split and baseline.\n3. Open Event replay, subscribe, and advance events.\n4. Point out a forecast before its actual destination is revealed.\n5. Unsubscribe, advance another event, and show that notifications stop.")
    st.subheader("Limitations to explain honestly")
    st.write("The source is a simulated network testbed with few devices. Registration timestamps are coarse; reports marked inactive are conservatively excluded. Cell changes are observed mobility proxies, not protocol-verified handovers. The app does not manage a live network, claim 3GPP compliance, or establish improved network performance.")
    st.caption("Dataset provenance, checksums, setup instructions and test methodology are included in the repository README and data/manifest.json.")

st.markdown('<div class="footnote">MINI NWDAF · Built for learning · Original testbed data, reproducible evaluation, clearly stated scope.</div>', unsafe_allow_html=True)
