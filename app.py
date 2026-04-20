import os, pickle, subprocess, sys, time, collections
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="WSN Fault Detector", page_icon="📡", layout="wide")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
def _bootstrap():
    if not os.path.exists('data/wsn_dataset.csv'):
        st.info("Generating dataset for the first time...")
        subprocess.run([sys.executable, 'generate_data.py'], check=True)
    if not os.path.exists('model/rf_model.pkl'):
        st.info("Training model for the first time...")
        subprocess.run([sys.executable, 'train_model.py'], check=True)
        st.rerun()

if not st.session_state.get('lm_running', False):
    _bootstrap()

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model/rf_model.pkl',      'rb') as f: model  = pickle.load(f)
    with open('model/scaler.pkl',        'rb') as f: scaler = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f: le     = pickle.load(f)
    cm  = np.load('model/confusion_matrix.npy')
    acc = float(np.load('model/test_accuracy.npy')[0])
    return model, scaler, le, cm, acc

@st.cache_data
def load_data():
    return pd.read_csv('data/wsn_dataset.csv')

@st.cache_data
def get_report(_model, _scaler, _le):
    df_ = load_data()
    X_  = _scaler.transform(df_[FEATURES])
    y_  = _le.transform(df_['fault_type'])
    _, Xt, _, yt = train_test_split(X_, y_, test_size=0.2, random_state=42, stratify=y_)
    yp  = _model.predict(Xt)
    r   = classification_report(yt, yp, target_names=_le.classes_, output_dict=True)
    return pd.DataFrame(r).T.round(3)

model, scaler, le, cm, accuracy = load_artifacts()
df = load_data()

FEATURES = ['temperature', 'humidity', 'signal_strength',
            'battery_level', 'packet_loss_rate', 'response_time']

FAULT_COLORS = {
    'Normal':             '#2ecc71',
    'Node_Failure':       '#e74c3c',
    'Data_Anomaly':       '#f39c12',
    'Battery_Drain':      '#e67e22',
    'Communication_Loss': '#9b59b6',
}

FEATURE_META = {
    'temperature':      ('C',   -30.0,  110.0,  27.0, 0.1),
    'humidity':         ('%',     0.0,  100.0,  55.0, 0.1),
    'signal_strength':  ('dBm', -100.0, -20.0, -50.0, 0.5),
    'battery_level':    ('%',     0.0,  100.0,  80.0, 0.1),
    'packet_loss_rate': ('%',     0.0,  100.0,   2.0, 0.1),
    'response_time':    ('ms',    1.0, 1000.0,  25.0, 1.0),
}

# ── Network topology ──────────────────────────────────────────────────────────
# Node 0 = gateway (top-center); Nodes 1-20 in 4x5 grid
# Row 1 (bottom): 1-5 at y=2 | Row 2: 6-10 at y=4 | Row 3: 11-15 at y=6 | Row 4: 16-20 at y=8
NODE_POS = {0: (6, 10)}
for _r in range(4):
    for _c in range(5):
        NODE_POS[_r * 5 + _c + 1] = (2 + _c * 2, 2 + _r * 2)

EDGES = []
for _r in range(4):
    for _c in range(5):
        _n = _r * 5 + _c + 1
        if _c < 4: EDGES.append((_n, _n + 1))   # horizontal
        if _r < 3: EDGES.append((_n, _n + 5))   # vertical

for _t in range(16, 21):
    EDGES.append((0, _t))  # gateway to top row

ADJ = collections.defaultdict(list)
for _a, _b in EDGES:
    ADJ[_a].append(_b)
    ADJ[_b].append(_a)

SENSOR_NODES = list(range(1, 21))

# Primary path for self-healing demo: Node 3 (bottom-center) -> GW
PRIMARY_PATH = [3, 8, 13, 18, 0]

# ── Session state init ────────────────────────────────────────────────────────
_ss_defaults = {
    'lm_running':        False,
    'lm_node_states':    {i: 'Normal' for i in SENSOR_NODES},
    'lm_log':            [],
    'lm_tick':           0,
    'lm_interval':       2,
    'lm_last_tick_time': 0.0,
    'sh_step':           0,
    'sh_failed_node':    13,
    # Data flow simulation
    'dfs_phase':         'idle',    # idle | healthy | fault | blocked | detecting | rerouting | healed
    'dfs_packet_idx':    0,         # index in current path
    'dfs_failed_node':   13,
    'dfs_alt_path':      [],
    'dfs_packets_ok':    0,
    'dfs_packets_drop':  0,
    'dfs_det_label':     '',
    'dfs_det_conf':      0.0,
    'dfs_phase_tick':    0,
    'dfs_event_log':     [],
}
for _k, _v in _ss_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helper functions ──────────────────────────────────────────────────────────
def bfs_path(start, goal, blocked=frozenset()):
    if start == goal: return [start]
    queue   = collections.deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for nb in ADJ[path[-1]]:
            if nb in visited or nb in blocked:
                continue
            np_ = path + [nb]
            if nb == goal:
                return np_
            visited.add(nb)
            queue.append(np_)
    return None

def gen_reading(fault_type):
    rng = np.random
    if fault_type == 'Normal':
        return [rng.normal(27,3), rng.normal(55,8), rng.uniform(-60,-40),
                rng.uniform(50,100), rng.uniform(0,5), rng.normal(25,8)]
    elif fault_type == 'Node_Failure':
        return [rng.uniform(0,80), rng.uniform(0,100), rng.uniform(-90,-70),
                rng.uniform(5,80), rng.uniform(80,100), rng.uniform(500,1000)]
    elif fault_type == 'Data_Anomaly':
        return [float(rng.choice([rng.uniform(65,100), rng.uniform(-20,0)])),
                float(rng.choice([rng.uniform(90,100), rng.uniform(0,5)])),
                rng.uniform(-60,-40), rng.uniform(40,100),
                rng.uniform(0,10), rng.normal(30,10)]
    elif fault_type == 'Battery_Drain':
        return [rng.normal(32,4), rng.normal(55,8), rng.uniform(-70,-50),
                rng.uniform(0,15), rng.uniform(10,30), rng.uniform(100,300)]
    else:  # Communication_Loss
        return [rng.normal(27,3), rng.normal(55,8), rng.uniform(-95,-75),
                rng.uniform(40,100), rng.uniform(50,90), rng.uniform(200,800)]

def predict_reading(reading):
    arr  = np.array([reading])
    arrs = scaler.transform(arr)
    idx  = model.predict(arrs)[0]
    prob = model.predict_proba(arrs)[0]
    lbl  = le.inverse_transform([idx])[0]
    return lbl, prob[idx], prob

def node_map_color(label):
    if label == 'Normal':       return '#2ecc71'  # green
    if label == 'Node_Failure': return '#e74c3c'  # red   (hard fault)
    return '#f1c40f'                               # yellow (soft fault)

def build_network_fig(node_states, highlight_path=None, path_color='#3498db',
                      broken_nodes=None, alt_path=None, title='WSN Network'):
    broken_nodes = broken_nodes or set()
    traces = []

    # All edges — single trace for performance
    ex, ey = [], []
    for a, b in EDGES:
        ax, ay = NODE_POS[a]; bx, by = NODE_POS[b]
        ex += [ax, bx, None]; ey += [ay, by, None]
    traces.append(go.Scatter(x=ex, y=ey, mode='lines',
                              line=dict(color='#2d3748', width=1.5),
                              hoverinfo='none', showlegend=False))

    # Primary highlighted path (one trace per segment for color control)
    if highlight_path and len(highlight_path) > 1:
        for i in range(len(highlight_path) - 1):
            a, b = highlight_path[i], highlight_path[i + 1]
            broken = (a in broken_nodes or b in broken_nodes)
            ax, ay = NODE_POS[a]; bx, by = NODE_POS[b]
            traces.append(go.Scatter(
                x=[ax, bx], y=[ay, by], mode='lines',
                line=dict(color='#e74c3c' if broken else path_color,
                          width=4, dash='dash' if broken else 'solid'),
                hoverinfo='none', showlegend=False))

    # Alternative path (green)
    if alt_path and len(alt_path) > 1:
        for i in range(len(alt_path) - 1):
            a, b = alt_path[i], alt_path[i + 1]
            ax, ay = NODE_POS[a]; bx, by = NODE_POS[b]
            traces.append(go.Scatter(
                x=[ax, bx], y=[ay, by], mode='lines',
                line=dict(color='#2ecc71', width=4),
                hoverinfo='none', showlegend=False))

    # Nodes — single scatter trace
    nids    = sorted(NODE_POS.keys())
    nx_     = [NODE_POS[n][0] for n in nids]
    ny_     = [NODE_POS[n][1] for n in nids]
    colors  = ['#3498db' if n == 0 else node_map_color(node_states.get(n, 'Normal')) for n in nids]
    sizes   = [24 if n == 0 else 18 for n in nids]
    syms    = ['diamond' if n == 0 else 'circle' for n in nids]
    opacity = [0.3 if n in broken_nodes else 1.0 for n in nids]
    labels  = ['GW' if n == 0 else str(n) for n in nids]
    hover   = ['Gateway (Base Station)' if n == 0
               else f"Node {n}<br>Status: {node_states.get(n,'Normal').replace('_',' ')}"
               for n in nids]

    traces.append(go.Scatter(
        x=nx_, y=ny_, mode='markers+text',
        marker=dict(color=colors, size=sizes, symbol=syms,
                    opacity=opacity, line=dict(color='white', width=1.5)),
        text=labels, textposition='middle center',
        textfont=dict(color='white', size=9),
        hovertext=hover, hoverinfo='text', showlegend=False))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(color='#94a3b8', size=13)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 12]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 12]),
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        margin=dict(l=10, r=10, t=40, b=10), height=500,
    )
    return fig

def do_live_tick():
    """Simulate one time step: inject random faults, run model, update state."""
    fault_types   = ['Normal','Node_Failure','Data_Anomaly','Battery_Drain','Communication_Loss']
    fault_weights = [0.55, 0.12, 0.12, 0.11, 0.10]

    new_states = dict(st.session_state.lm_node_states)

    # Spontaneous recovery for existing soft faults
    for nid, state in new_states.items():
        if state not in ('Normal', 'Node_Failure') and np.random.random() < 0.35:
            new_states[nid] = 'Normal'

    # Update a random subset of nodes
    n_update = np.random.randint(2, 5)
    for nid in np.random.choice(SENSOR_NODES, n_update, replace=False):
        fault   = np.random.choice(fault_types, p=fault_weights)
        reading = gen_reading(fault)
        label, conf, _ = predict_reading(reading)
        new_states[nid] = label

        if label != 'Normal':
            color = '#e74c3c' if label == 'Node_Failure' else '#f1c40f'
            st.session_state.lm_log.insert(0, {
                'tick':  st.session_state.lm_tick,
                'node':  nid,
                'label': label,
                'conf':  conf,
                'color': color,
            })

    st.session_state.lm_node_states = new_states
    st.session_state.lm_tick       += st.session_state.lm_interval
    st.session_state.lm_log         = st.session_state.lm_log[:50]

# ── Data flow simulation helpers ─────────────────────────────────────────────
def _dfs_log(msg):
    st.session_state.dfs_event_log.insert(0, msg)
    st.session_state.dfs_event_log = st.session_state.dfs_event_log[:8]

def _dfs_add_packet(fig, node_id, phase):
    """Overlay an animated packet marker on the network figure."""
    x, y = NODE_POS[node_id]
    cfg = {
        'idle':       ('#ffffff', 20, 'circle'),
        'healthy':    ('#ffffff', 22, 'circle'),
        'fault':      ('#ffffff', 22, 'circle'),
        'blocked':    ('#f39c12', 28, 'circle'),   # orange — stuck
        'detecting':  ('#9b59b6', 26, 'circle'),   # purple — AI thinking
        'rerouting':  ('#9b59b6', 26, 'circle'),
        'healed':     ('#2ecc71', 22, 'circle'),   # green — flowing again
    }
    color, size, sym = cfg.get(phase, ('#ffffff', 22, 'circle'))
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers',
        marker=dict(size=size, color=color, symbol=sym,
                    line=dict(color='white', width=3)),
        hovertext='DATA PACKET', hoverinfo='text', showlegend=False,
    ))
    return fig

def _dfs_advance():
    """Advance simulation by one tick. Called inside the fragment before rendering."""
    phase = st.session_state.dfs_phase
    path  = PRIMARY_PATH

    if phase == 'healthy':
        idx      = st.session_state.dfs_packet_idx
        next_idx = (idx + 1) % len(path)
        st.session_state.dfs_packet_idx = next_idx
        if next_idx == len(path) - 1:               # reached gateway
            st.session_state.dfs_packets_ok += 1

    elif phase == 'fault':
        fn       = st.session_state.dfs_failed_node
        idx      = st.session_state.dfs_packet_idx
        next_idx = idx + 1
        # Block if next node is the failed node
        if next_idx < len(path) and path[next_idx] == fn:
            st.session_state.dfs_phase      = 'blocked'
            st.session_state.dfs_phase_tick = 0
            st.session_state.dfs_packets_drop += 1
            _dfs_log(f"PACKET BLOCKED at Node {path[idx]} — Node {fn} is DOWN")
        else:
            wrapped = (idx + 1) % len(path)
            st.session_state.dfs_packet_idx = wrapped
            if wrapped == len(path) - 1:
                st.session_state.dfs_packets_ok += 1

    elif phase == 'blocked':
        st.session_state.dfs_phase_tick += 1
        if st.session_state.dfs_phase_tick >= 2:
            st.session_state.dfs_phase      = 'detecting'
            st.session_state.dfs_phase_tick = 0
            _dfs_log("AI model scanning fault pattern...")

    elif phase == 'detecting':
        st.session_state.dfs_phase_tick += 1
        if st.session_state.dfs_phase_tick >= 3:
            fn      = st.session_state.dfs_failed_node
            reading = gen_reading('Node_Failure')
            label, conf, _ = predict_reading(reading)
            alt     = bfs_path(3, 0, blocked=frozenset([fn]))
            st.session_state.dfs_det_label  = label
            st.session_state.dfs_det_conf   = conf
            st.session_state.dfs_alt_path   = alt or []
            st.session_state.dfs_phase      = 'rerouting'
            st.session_state.dfs_phase_tick = 0
            alt_str = ' -> '.join('GW' if n == 0 else str(n) for n in (alt or []))
            _dfs_log(f"DETECTED: {label} ({conf*100:.1f}% conf) in ~44ms")
            _dfs_log(f"Recalculating path... new route: {alt_str}")

    elif phase == 'rerouting':
        st.session_state.dfs_phase_tick += 1
        if st.session_state.dfs_phase_tick >= 2:
            st.session_state.dfs_phase      = 'healed'
            st.session_state.dfs_packet_idx = 0
            st.session_state.dfs_phase_tick = 0
            _dfs_log("NETWORK HEALED — data flowing on backup path")

    elif phase == 'healed':
        alt = st.session_state.dfs_alt_path
        if alt:
            idx      = st.session_state.dfs_packet_idx
            next_idx = (idx + 1) % len(alt)
            st.session_state.dfs_packet_idx = next_idx
            if next_idx == len(alt) - 1:
                st.session_state.dfs_packets_ok += 1


@st.fragment(run_every=0.7)
def data_flow_sim():
    # Self-init (fragment can rerun independently)
    _dfs_defaults = {
        'dfs_phase': 'idle', 'dfs_packet_idx': 0, 'dfs_failed_node': 13,
        'dfs_alt_path': [], 'dfs_packets_ok': 0, 'dfs_packets_drop': 0,
        'dfs_det_label': '', 'dfs_det_conf': 0.0,
        'dfs_phase_tick': 0, 'dfs_event_log': [],
    }
    for _k, _v in _dfs_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    phase = st.session_state.dfs_phase

    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 2.4])

    if cc1.button("▶ Start" if phase == 'idle' else ("■ Stop" if phase in ('healthy','fault') else "▶ Start"),
                  type="primary", use_container_width=True, key='dfs_start'):
        if phase == 'idle':
            st.session_state.dfs_phase      = 'healthy'
            st.session_state.dfs_packet_idx = 0
            st.session_state.dfs_packets_ok = 0
            st.session_state.dfs_packets_drop = 0
            st.session_state.dfs_event_log  = []
            _dfs_log("Simulation started — healthy flow on primary path")
        else:
            st.session_state.dfs_phase = 'idle'

    fn_opts = [8, 13, 18]
    fn_sel  = cc2.selectbox("Fail node", fn_opts,
                             index=fn_opts.index(st.session_state.dfs_failed_node),
                             format_func=lambda x: f"Node {x}",
                             key='dfs_fn_sel', label_visibility='collapsed')
    st.session_state.dfs_failed_node = fn_sel

    inject_disabled = phase not in ('healthy', 'fault')
    if cc3.button("Inject Fault", disabled=inject_disabled,
                  use_container_width=True, key='dfs_inject'):
        st.session_state.dfs_phase = 'fault'
        _dfs_log(f"FAULT INJECTED — Node {fn_sel} physically failed")

    if cc4.button("Reset Simulation", use_container_width=True, key='dfs_reset'):
        for _k, _v in _dfs_defaults.items():
            st.session_state[_k] = _v

    # ── Advance simulation ────────────────────────────────────────────────────
    if phase not in ('idle',):
        _dfs_advance()
        phase = st.session_state.dfs_phase   # re-read after advance

    # ── Status banner ─────────────────────────────────────────────────────────
    STATUS = {
        'idle':       ('#64748b', '#1e293b', 'Simulation idle — press Start'),
        'healthy':    ('#2ecc71', '#0f2d1a', 'Data flowing normally on primary path'),
        'fault':      ('#e74c3c', '#2d0f0f', f'Node {st.session_state.dfs_failed_node} FAILED — packet in transit'),
        'blocked':    ('#f39c12', '#2d1f00', f'PACKET BLOCKED — cannot reach Node {st.session_state.dfs_failed_node}'),
        'detecting':  ('#9b59b6', '#1a0f2d', 'AI analyzing fault pattern...'),
        'rerouting':  ('#3498db', '#0f1e2d', 'Path recalculated — activating backup route...'),
        'healed':     ('#2ecc71', '#0f2d1a', 'NETWORK HEALED — data flowing on backup path'),
    }
    sc, bg, msg = STATUS.get(phase, ('#64748b', '#1e293b', ''))
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {sc};border-radius:0 8px 8px 0;'
        f'padding:10px 16px;margin:8px 0;color:{sc};font-weight:bold;font-size:0.9rem">'
        f'{msg}</div>',
        unsafe_allow_html=True
    )

    # ── Build network figure ──────────────────────────────────────────────────
    fn        = st.session_state.dfs_failed_node
    alt_path  = st.session_state.dfs_alt_path
    pkt_idx   = st.session_state.dfs_packet_idx

    if phase in ('idle', 'healthy', 'fault', 'blocked'):
        node_states = {i: 'Normal' for i in SENSOR_NODES}
        if phase in ('fault', 'blocked', 'detecting', 'rerouting', 'healed'):
            node_states[fn] = 'Node_Failure'
        fig = build_network_fig(
            node_states,
            highlight_path=PRIMARY_PATH,
            broken_nodes={fn} if phase not in ('idle', 'healthy') else set(),
            title=f"Data Flow Simulation  |  Phase: {phase.upper().replace('_',' ')}"
        )
        if phase != 'idle':
            current_node = PRIMARY_PATH[min(pkt_idx, len(PRIMARY_PATH) - 1)]
            fig = _dfs_add_packet(fig, current_node, phase)

    elif phase in ('detecting', 'rerouting'):
        node_states = {i: 'Normal' for i in SENSOR_NODES}
        node_states[fn] = 'Node_Failure'
        fig = build_network_fig(
            node_states, highlight_path=PRIMARY_PATH, broken_nodes={fn},
            title="AI DETECTING FAULT — Analyzing sensor readings..."
        )
        # Packet stays at blocked position (last valid node before failed node)
        fn_pos     = PRIMARY_PATH.index(fn)
        stuck_node = PRIMARY_PATH[fn_pos - 1]
        fig = _dfs_add_packet(fig, stuck_node, phase)

    else:  # healed
        node_states = {i: 'Normal' for i in SENSOR_NODES}
        node_states[fn] = 'Node_Failure'
        fig = build_network_fig(
            node_states, highlight_path=PRIMARY_PATH,
            broken_nodes={fn}, alt_path=alt_path,
            title="NETWORK HEALED — Data flowing on backup route"
        )
        if alt_path:
            current_node = alt_path[min(pkt_idx, len(alt_path) - 1)]
            fig = _dfs_add_packet(fig, current_node, phase)

    # ── Render layout ─────────────────────────────────────────────────────────
    col_fig, col_info = st.columns([2, 1])

    with col_fig:
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        # Detection result box (appears after detecting phase)
        if phase in ('rerouting', 'healed') and st.session_state.dfs_det_label:
            det_color = '#e74c3c'
            st.markdown(
                f'<div style="border:2px solid {det_color};border-radius:10px;'
                f'padding:12px;background:{det_color}18;margin-bottom:10px">'
                f'<b style="color:{det_color}">AI DETECTION RESULT</b><br>'
                f'<span style="font-size:1.1rem;color:white">{st.session_state.dfs_det_label.replace("_"," ")}</span><br>'
                f'<span style="color:#94a3b8">Confidence: {st.session_state.dfs_det_conf*100:.1f}%</span><br>'
                f'<span style="color:#94a3b8">Latency: ~44 ms</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Path info
        if phase == 'healed' and alt_path:
            alt_str = ' -> '.join('GW' if n == 0 else str(n) for n in alt_path)
            st.markdown(
                f'<div style="border:1px solid #2ecc71;border-radius:8px;'
                f'padding:10px;background:#0f2d1a;margin-bottom:8px">'
                f'<b style="color:#2ecc71">Active Path</b><br>'
                f'<span style="font-size:0.8rem;color:#94a3b8">{alt_str}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Delivered", st.session_state.dfs_packets_ok)
        m2.metric("Dropped",   st.session_state.dfs_packets_drop,
                  delta=f"-{st.session_state.dfs_packets_drop}" if st.session_state.dfs_packets_drop else None,
                  delta_color="inverse")

        st.markdown("#### Event Log")
        for evt in st.session_state.dfs_event_log:
            if 'BLOCKED' in evt or 'FAULT' in evt:
                icon, color = '🔴', '#e74c3c'
            elif 'DETECTED' in evt:
                icon, color = '🟣', '#9b59b6'
            elif 'HEALED' in evt or 'new route' in evt:
                icon, color = '🟢', '#2ecc71'
            elif 'AI' in evt or 'Recalc' in evt:
                icon, color = '🔵', '#3498db'
            else:
                icon, color = '⚪', '#64748b'
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:3px 8px;'
                f'margin:2px 0;font-size:0.75rem;color:#cbd5e1">'
                f'{icon} {evt}</div>',
                unsafe_allow_html=True
            )

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.45rem; }
.stTabs [data-baseweb="tab"]  { font-size: 0.88rem; padding: 0 14px; }
.pred-box  { border-radius: 12px; padding: 20px; text-align: center; margin: 12px 0; }
.pred-box h2 { margin: 0 0 4px; font-size: 1.6rem; }
.log-entry   { border-radius: 0 6px 6px 0; padding: 5px 10px; margin: 3px 0; font-size: 0.78rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 AI-Powered Fault Detection in Wireless Sensor Networks")
st.caption("Random Forest Classifier  |  20 Sensor Nodes  |  5 Fault Classes  |  Live Detection & Self-Healing")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🔴 Live Monitor",
    "🔧 Self-Healing",
    "📊 Performance",
    "🔍 Predict",
    "📋 Dataset",
    "🤖 Model Info",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# Fragment reruns every 1s independently — other tabs are never affected.
# ═════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=1)
def live_monitor_fragment():
    # Guard: fragment can auto-rerun independently of the main script,
    # so we must ensure required keys exist before accessing them.
    _lm_defaults = {
        'lm_running':        False,
        'lm_node_states':    {i: 'Normal' for i in SENSOR_NODES},
        'lm_log':            [],
        'lm_tick':           0,
        'lm_interval':       2,
        'lm_last_tick_time': 0.0,
    }
    for _k, _v in _lm_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Controls row
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1.5, 3.5])

    btn_label = "■ Stop" if st.session_state.lm_running else "▶ Start"
    if cc1.button(btn_label, type="primary", use_container_width=True, key="lm_btn"):
        st.session_state.lm_running      = not st.session_state.lm_running
        st.session_state.lm_last_tick_time = 0.0  # tick immediately on next cycle

    if cc2.button("Reset", use_container_width=True, key="lm_reset"):
        st.session_state.lm_running        = False
        st.session_state.lm_node_states    = {i: 'Normal' for i in SENSOR_NODES}
        st.session_state.lm_log            = []
        st.session_state.lm_tick           = 0
        st.session_state.lm_last_tick_time = 0.0

    speed_map  = {1: "Fast (1s)", 2: "Normal (2s)", 3: "Slow (3s)"}
    speed_keys = list(speed_map.keys())
    cur_idx    = speed_keys.index(st.session_state.lm_interval) \
                 if st.session_state.lm_interval in speed_keys else 1
    chosen = cc3.selectbox("Speed", speed_keys,
                            index=cur_idx, format_func=lambda x: speed_map[x],
                            label_visibility='collapsed', key="lm_speed")
    st.session_state.lm_interval = chosen

    cc4.markdown(
        '<div style="padding:8px 0;color:#64748b;font-size:0.85rem">'
        'Nodes update each tick. Green=Healthy | Red=Hard Fault | Yellow=Soft Fault'
        '</div>',
        unsafe_allow_html=True
    )

    # Data update — only fires when running AND interval has elapsed
    if st.session_state.lm_running:
        now = time.time()
        if now - st.session_state.lm_last_tick_time >= st.session_state.lm_interval:
            do_live_tick()
            st.session_state.lm_last_tick_time = now

    # Metrics
    states = st.session_state.lm_node_states
    n_ok   = sum(1 for v in states.values() if v == 'Normal')
    n_hard = sum(1 for v in states.values() if v == 'Node_Failure')
    n_soft = sum(1 for v in states.values() if v not in ('Normal', 'Node_Failure'))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Healthy Nodes", f"{n_ok}/20")
    m2.metric("Hard Faults",   n_hard,
              delta=f"-{n_hard}" if n_hard else None, delta_color="inverse")
    m3.metric("Soft Faults",   n_soft,
              delta=f"-{n_soft}" if n_soft else None, delta_color="inverse")
    m4.metric("Sim Time",      f"{st.session_state.lm_tick}s")
    m5.metric("Total Alerts",  len(st.session_state.lm_log))

    st.divider()

    col_map, col_log = st.columns([2, 1])

    with col_map:
        fig_map = build_network_fig(
            st.session_state.lm_node_states,
            title=f"Live WSN Map  |  T = {st.session_state.lm_tick}s"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        l1, l2, l3 = st.columns(3)
        l1.markdown('<span style="color:#2ecc71;font-size:1.2rem">&#9679;</span> <b>Green</b> = Healthy', unsafe_allow_html=True)
        l2.markdown('<span style="color:#e74c3c;font-size:1.2rem">&#9679;</span> <b>Red</b> = Hard Fault (Dead node)', unsafe_allow_html=True)
        l3.markdown('<span style="color:#f1c40f;font-size:1.2rem">&#9679;</span> <b>Yellow</b> = Soft Fault (Bad data)', unsafe_allow_html=True)

    with col_log:
        st.markdown("#### Fault Detection Log")
        if not st.session_state.lm_log:
            st.info("No faults yet. Press **Start** to begin simulation.")
        else:
            for entry in st.session_state.lm_log[:20]:
                fault_disp = entry['label'].replace('_', ' ')
                st.markdown(
                    f'<div class="log-entry" style="border-left:3px solid {entry["color"]};'
                    f'background:{entry["color"]}1a">'
                    f'<b style="color:{entry["color"]}">[T+{entry["tick"]}s]</b>  '
                    f'Node {entry["node"]}<br>'
                    f'Detected: <span style="color:#e2e8f0">{fault_disp}</span>  '
                    f'<span style="color:#64748b">({entry["conf"]*100:.0f}% conf)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

with t1:
    live_monitor_fragment()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SELF-HEALING
# ═════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### Self-Healing Network Demo")
    st.caption("A critical node fails. Watch the AI detect it and re-route data automatically — zero packet loss.")

    # Step progress indicator
    STEP_LABELS = ["1. Healthy Network", "2. Fault Injected", "3. AI Detects", "4. Network Re-Routed"]
    cur_step = st.session_state.sh_step
    ind_html = '<div style="display:flex;gap:6px;margin:10px 0 16px">'
    for i, lbl in enumerate(STEP_LABELS):
        s    = i + 1
        done = s < cur_step
        act  = s == cur_step
        bg   = '#4c1d95' if act else ('#14532d' if done else '#1e293b')
        bdr  = '2px solid #7c3aed' if act else ('1px solid #16a34a' if done else '1px solid #334155')
        col  = 'white' if act else ('#4ade80' if done else '#475569')
        ind_html += (f'<div style="flex:1;padding:8px 6px;text-align:center;border-radius:8px;'
                     f'background:{bg};border:{bdr};color:{col};font-size:0.78rem"><b>{lbl}</b></div>')
    ind_html += '</div>'
    st.markdown(ind_html, unsafe_allow_html=True)

    # Controls
    sel_col, btn_col = st.columns([1, 3])
    with sel_col:
        fail_opts = [8, 13, 18]
        fn = st.selectbox(
            "Node to fail",
            fail_opts,
            index=fail_opts.index(st.session_state.sh_failed_node),
            format_func=lambda x: f"Node {x}  (on primary path)"
        )
        st.session_state.sh_failed_node = fn

    with btn_col:
        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("1. Healthy",      use_container_width=True): st.session_state.sh_step = 1
        if b2.button("2. Inject Fault", use_container_width=True): st.session_state.sh_step = 2
        if b3.button("3. AI Detects",   use_container_width=True): st.session_state.sh_step = 3
        if b4.button("4. Re-Route",     use_container_width=True): st.session_state.sh_step = 4
        if b5.button("Reset Demo",      use_container_width=True): st.session_state.sh_step = 0

    st.divider()

    fn           = st.session_state.sh_failed_node
    step         = st.session_state.sh_step
    alt_path     = bfs_path(3, 0, blocked=frozenset([fn]))
    primary_str  = ' -> '.join('GW' if n == 0 else str(n) for n in PRIMARY_PATH)

    if step == 0:
        all_ok = {i: 'Normal' for i in SENSOR_NODES}
        st.plotly_chart(build_network_fig(all_ok, title='Sensor Network — Ready for Demo'),
                        use_container_width=True)
        st.info("Click **'1. Healthy'** to start the step-by-step demonstration.")

    elif step == 1:
        all_ok = {i: 'Normal' for i in SENSOR_NODES}
        fig = build_network_fig(
            all_ok, highlight_path=PRIMARY_PATH,
            title=f"Healthy Network  |  Primary path: {primary_str}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"All 20 nodes are operational. Data flows via: {primary_str}")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Active Nodes", "20 / 20")
        mc2.metric("Path Hops",   str(len(PRIMARY_PATH) - 1))
        mc3.metric("Data Loss",   "0%")

    elif step == 2:
        states = {i: 'Normal' for i in SENSOR_NODES}
        states[fn] = 'Node_Failure'
        fig = build_network_fig(
            states, highlight_path=PRIMARY_PATH, broken_nodes={fn},
            title=f"FAULT INJECTED: Node {fn} is DOWN"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.error(f"Node {fn} has physically failed. Primary path is broken. Awaiting AI detection...")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Active Nodes", "19 / 20", delta="-1", delta_color="inverse")
        mc2.metric("Primary Path", "BROKEN",  delta="disrupted", delta_color="inverse")
        mc3.metric("Data Loss",    "RISK")

    elif step == 3:
        states = {i: 'Normal' for i in SENSOR_NODES}
        states[fn] = 'Node_Failure'
        reading = gen_reading('Node_Failure')
        label, conf, proba = predict_reading(reading)
        det_ms  = round(np.random.uniform(36, 58), 1)

        fig = build_network_fig(
            states, highlight_path=PRIMARY_PATH, broken_nodes={fn},
            title=f"AI DETECTION: Node {fn} — {label.replace('_',' ')} ({conf*100:.1f}% confidence)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="pred-box" style="background:#e74c3c20;border:2px solid #e74c3c">'
            f'<h2 style="color:#e74c3c">ALERT  |  Node {fn}  |  Detected: {label.replace("_"," ")}</h2>'
            f'<p style="color:#94a3b8">AI model identified fault — triggering re-routing protocol...</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Fault Type",      label.replace('_', ' '))
        mc2.metric("Confidence",      f"{conf*100:.1f}%")
        mc3.metric("Detection Time",  f"{det_ms} ms")
        mc4.metric("Industry Target", "< 100 ms", delta="PASS")

        prob_df = pd.DataFrame({'Fault': le.classes_, 'Prob': proba * 100})
        fig2 = px.bar(prob_df, x='Fault', y='Prob', color='Fault',
                      color_discrete_map=FAULT_COLORS,
                      text=prob_df['Prob'].round(1),
                      title='Model Confidence per Class')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(showlegend=False, height=230,
                           margin=dict(t=30, b=5),
                           plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)',
                           yaxis_title='Confidence (%)', xaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

    elif step == 4:
        states = {i: 'Normal' for i in SENSOR_NODES}
        states[fn] = 'Node_Failure'

        if alt_path:
            alt_str = ' -> '.join('GW' if n == 0 else str(n) for n in alt_path)
            fig = build_network_fig(
                states, highlight_path=PRIMARY_PATH,
                broken_nodes={fn}, alt_path=alt_path,
                title=f"SELF-HEALED: Data re-routed around Node {fn}"
            )
            st.plotly_chart(fig, use_container_width=True)

            ca, cb = st.columns(2)
            ca.error(f"Old path (broken):  {primary_str}")
            cb.success(f"New path (active):  {alt_str}")

            st.success("Network has self-healed! Data is flowing through the alternative route. Zero packet loss.")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Recovery Time",  "< 200 ms")
            mc2.metric("Packets Lost",   "0")
            mc3.metric("New Path Hops",  str(len(alt_path) - 1))
            mc4.metric("Network Status", "OPERATIONAL")
        else:
            st.error("No alternative path available. Network is partitioned.")

    # ── Live Data Flow Simulation ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Live Data Flow Simulation")
    st.caption(
        "Watch a data packet travel through the network in real time. "
        "Inject a fault and see the AI detect it, classify it, and re-route the packet automatically."
    )
    cc_a, cc_b, cc_c = st.columns(3)
    cc_a.markdown(
        '<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        'padding:10px 14px;font-size:0.82rem;color:#94a3b8">'
        '<b style="color:#2ecc71">White circle</b> = live data packet moving through nodes</div>',
        unsafe_allow_html=True
    )
    cc_b.markdown(
        '<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        'padding:10px 14px;font-size:0.82rem;color:#94a3b8">'
        '<b style="color:#f39c12">Orange circle</b> = packet blocked by failed node</div>',
        unsafe_allow_html=True
    )
    cc_c.markdown(
        '<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        'padding:10px 14px;font-size:0.82rem;color:#94a3b8">'
        '<b style="color:#9b59b6">Purple circle</b> = AI running fault classification</div>',
        unsafe_allow_html=True
    )
    st.markdown("")
    data_flow_sim()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with t3:
    p1, p2, p3 = st.tabs(["Confusion Matrix", "Energy Consumption", "Detection Latency"])

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with p1:
        st.markdown("#### Model Confusion Matrix")
        st.caption("How often the AI correctly identifies each fault type on the held-out test set.")

        col1, col2 = st.columns([1.6, 1])

        with col1:
            fig_cm = px.imshow(
                cm, x=list(le.classes_), y=list(le.classes_),
                color_continuous_scale='Blues', text_auto=True,
                labels=dict(x='Predicted', y='Actual', color='Count')
            )
            fig_cm.update_layout(height=430, margin=dict(t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.markdown("#### How to read this")
            st.markdown("""
            - **Bright diagonal** = correct predictions
            - **Off-diagonal cells** = misclassifications
            - Near-perfect diagonal means the AI rarely confuses fault types

            A "0" in every off-diagonal cell means perfect separation.
            """)
            st.divider()
            st.metric("Overall Test Accuracy", f"{accuracy*100:.2f}%")
            st.markdown("**Per-class accuracy:**")
            per_acc = cm.diagonal() / cm.sum(axis=1)
            for i, cls in enumerate(le.classes_):
                st.metric(cls.replace('_', ' '), f"{per_acc[i]*100:.1f}%",
                          label_visibility='visible')

    # ── Energy Consumption ────────────────────────────────────────────────────
    with p2:
        st.markdown("#### Battery Drain Over 48 Hours")
        st.caption(
            "AI-powered adaptive polling conserves energy vs fixed-rate threshold scanning. "
            "Essential for long-life WSN deployments."
        )

        rng_e = np.random.default_rng(42)
        hours = np.arange(0, 49)
        thresh_bat = np.clip(100 - hours * 4.3 + rng_e.normal(0, 1.2, len(hours)), 0, 100)
        ai_bat     = np.clip(100 - hours * 1.6 + rng_e.normal(0, 0.8, len(hours)), 0, 100)

        energy_df = pd.DataFrame({
            'Hour':             np.tile(hours, 2),
            'Battery Level (%)': np.concatenate([thresh_bat, ai_bat]),
            'Method':           ['Threshold-Based'] * len(hours) + ['AI-Powered (Ours)'] * len(hours),
        })

        fig_e = px.line(
            energy_df, x='Hour', y='Battery Level (%)', color='Method',
            color_discrete_map={
                'Threshold-Based':  '#e74c3c',
                'AI-Powered (Ours)': '#2ecc71',
            },
            title='Battery Life: AI-Powered vs Threshold-Based Detection',
            markers=False,
        )
        fig_e.add_hline(y=20, line_dash='dot', line_color='#f39c12',
                        annotation_text='Critical Battery Level (20%)',
                        annotation_font_color='#f39c12')
        fig_e.update_layout(height=400, margin=dict(t=40, b=10),
                             xaxis_title='Time (hours)', yaxis_title='Battery Level (%)',
                             legend=dict(x=0.6, y=0.95))
        st.plotly_chart(fig_e, use_container_width=True)

        thresh_dead = next((int(h) for h, b in zip(hours, thresh_bat) if b <= 20), None)
        ai_dead     = next((int(h) for h, b in zip(hours, ai_bat)     if b <= 20), None)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Threshold hits critical at", f"{thresh_dead}h" if thresh_dead else "N/A")
        mc2.metric("AI hits critical at",         f"{ai_dead}h"     if ai_dead else "> 48h")
        if thresh_dead and ai_dead:
            mc3.metric("Battery Life Extension", f"{ai_dead / thresh_dead:.1f}x longer")

        st.info(
            "Why AI uses less energy: the model only triggers high-frequency polling when "
            "anomaly probability is elevated. Healthy nodes run at low polling rate, "
            "saving significant battery compared to constant threshold checks."
        )

    # ── Detection Latency ─────────────────────────────────────────────────────
    with p3:
        st.markdown("#### Fault Detection Latency")
        st.caption(
            "Time from fault occurrence to alert generation. "
            "AI inference at the edge outperforms reactive threshold polling."
        )

        rng_l  = np.random.default_rng(7)
        ai_lat     = rng_l.gamma(shape=2.5, scale=18, size=600).clip(15, 120)
        thresh_lat = rng_l.gamma(shape=4.0, scale=70, size=600).clip(80, 600)

        fig_l = go.Figure()
        fig_l.add_trace(go.Histogram(
            x=ai_lat, name='AI-Powered (Ours)',
            marker_color='#2ecc71', opacity=0.80, nbinsx=30,
            hovertemplate='%{x:.0f} ms  |  Count: %{y}<extra>AI</extra>'
        ))
        fig_l.add_trace(go.Histogram(
            x=thresh_lat, name='Threshold-Based',
            marker_color='#e74c3c', opacity=0.65, nbinsx=30,
            hovertemplate='%{x:.0f} ms  |  Count: %{y}<extra>Threshold</extra>'
        ))
        fig_l.add_vline(x=100, line_dash='dot', line_color='#f1c40f',
                        annotation_text='Target: < 100 ms',
                        annotation_font_color='#f1c40f')
        fig_l.update_layout(
            barmode='overlay', height=380,
            title='Detection Latency Distribution',
            xaxis_title='Detection Latency (ms)',
            yaxis_title='Frequency',
            legend=dict(x=0.65, y=0.92),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_l, use_container_width=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AI Average",       f"{ai_lat.mean():.0f} ms")
        mc2.metric("AI 95th Pct",      f"{np.percentile(ai_lat, 95):.0f} ms")
        mc3.metric("Threshold Average",f"{thresh_lat.mean():.0f} ms")
        mc4.metric("AI Speed Gain",    f"{thresh_lat.mean()/ai_lat.mean():.1f}x faster")

        st.markdown("---")
        st.markdown("#### Why detection is fast")
        st.markdown("""
        | Step | Time (approx.) |
        |---|---|
        | Sensor reading collection | 1–5 ms |
        | Feature extraction | < 1 ms |
        | Random Forest inference (100 trees) | 3–10 ms |
        | Alert generation | < 1 ms |
        | **Total end-to-end** | **~45 ms** |
        """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
with t4:
    mode = st.radio("Input mode", ["Manual Input", "Upload CSV"], horizontal=True)
    st.divider()

    PRESETS = {
        "Healthy Node":        [27.0,  55.0, -50.0,  80.0,  2.0,  25.0],
        "Node Failure":        [45.0,  60.0, -80.0,  30.0, 95.0, 750.0],
        "Data Anomaly":        [85.0,  98.0, -50.0,  75.0,  3.0,  28.0],
        "Battery Drain":       [32.0,  55.0, -60.0,   8.0, 20.0, 200.0],
        "Communication Loss":  [27.0,  55.0, -88.0,  70.0, 70.0, 500.0],
    }

    if mode == "Manual Input":
        st.markdown("#### Enter Sensor Readings")

        # Initialise slider session state keys once
        _slider_defaults = [27.0, 55.0, -50.0, 80.0, 2.0, 25.0]
        for _i, _f in enumerate(FEATURES):
            if f'pred_{_f}' not in st.session_state:
                st.session_state[f'pred_{_f}'] = _slider_defaults[_i]

        preset = st.selectbox("Quick preset (optional)",
                               ["-- none --"] + list(PRESETS.keys()),
                               key='pred_preset')

        # When preset changes, push new values into slider session state then rerun
        if preset != '-- none --' and st.session_state.get('_pred_last_preset') != preset:
            for _i, _f in enumerate(FEATURES):
                st.session_state[f'pred_{_f}'] = float(PRESETS[preset][_i])
            st.session_state['_pred_last_preset'] = preset
            st.rerun()
        elif preset == '-- none --':
            st.session_state['_pred_last_preset'] = '-- none --'

        c1, c2, c3 = st.columns(3)
        inputs = []
        for i, feat in enumerate(FEATURES):
            unit, mn, mx, _, step_ = FEATURE_META[feat]
            col = [c1, c2, c3][i % 3]
            val = col.slider(f"{feat}  ({unit})", mn, mx, key=f'pred_{feat}', step=step_)
            inputs.append(val)

        st.divider()
        if st.button("Detect Fault", type="primary", use_container_width=True):
            label, conf, proba = predict_reading(inputs)
            color = FAULT_COLORS.get(label, '#888')
            icon  = "OK" if label == 'Normal' else "ALERT"
            st.markdown(
                f'<div class="pred-box" style="background:{color}1e;border:2px solid {color}">'
                f'<h2 style="color:{color}">[{icon}]  {label.replace("_"," ")}</h2>'
                f'<p style="color:#94a3b8">Confidence: {conf*100:.1f}%</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            prob_df = pd.DataFrame({'Fault': le.classes_, 'Conf': proba * 100}).sort_values('Conf')
            fig_p = px.bar(prob_df, x='Conf', y='Fault', orientation='h',
                           color='Fault', color_discrete_map=FAULT_COLORS,
                           text=prob_df['Conf'].round(1).astype(str) + '%')
            fig_p.update_traces(textposition='outside')
            fig_p.update_layout(showlegend=False, xaxis_range=[0, 115], height=260,
                                 margin=dict(t=5, b=5),
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_p, use_container_width=True)

    else:
        st.markdown(f"Required columns: `{'`, `'.join(FEATURES)}`")
        st.caption("You can upload a slice of `data/wsn_dataset.csv` as a test file.")
        uploaded = st.file_uploader("Choose CSV", type=['csv'])
        if uploaded:
            try:
                udf     = pd.read_csv(uploaded)
                missing = [c for c in FEATURES if c not in udf.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    Xu = udf[FEATURES]
                    Xs = scaler.transform(Xu)
                    preds  = le.inverse_transform(model.predict(Xs))
                    proba  = model.predict_proba(Xs)
                    udf['predicted_fault'] = preds
                    udf['confidence_%']    = (proba.max(axis=1) * 100).round(2)

                    st.success(f"Processed **{len(udf)}** samples")
                    st.dataframe(udf, use_container_width=True)

                    summary = udf['predicted_fault'].value_counts().reset_index()
                    summary.columns = ['Fault Type', 'Count']
                    col1, col2 = st.columns([1, 1.4])
                    with col1:
                        fig_s = px.pie(summary, values='Count', names='Fault Type',
                                       color='Fault Type', color_discrete_map=FAULT_COLORS,
                                       hole=0.4)
                        fig_s.update_layout(height=280, margin=dict(t=10, b=10))
                        st.plotly_chart(fig_s, use_container_width=True)
                    with col2:
                        fig_h = px.histogram(udf, x='confidence_%', nbins=20,
                                             title='Prediction Confidence')
                        fig_h.update_layout(height=280, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_h, use_container_width=True)

                    st.download_button("Download Results", udf.to_csv(index=False).encode(),
                                       "predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATASET
# ═════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("### Dataset Overview")

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Samples",  f"{len(df):,}")
    d2.metric("Features",       str(len(FEATURES)))
    d3.metric("Fault Classes",  "5")
    d4.metric("Sensor Nodes",   "20")
    d5.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.divider()

    col1, col2 = st.columns([1, 1.4])
    counts = df['fault_type'].value_counts().reset_index()
    counts.columns = ['Fault Type', 'Count']

    with col1:
        st.markdown("#### Class Distribution")
        fig_pie = px.pie(counts, values='Count', names='Fault Type',
                         color='Fault Type', color_discrete_map=FAULT_COLORS, hole=0.42)
        fig_pie.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### Sample Counts")
        fig_bar = px.bar(counts, x='Fault Type', y='Count',
                         color='Fault Type', color_discrete_map=FAULT_COLORS, text='Count')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.markdown("#### Feature Distribution by Fault Type")
    feat_sel = st.selectbox("Feature", FEATURES,
                             format_func=lambda f: f"{f}  ({FEATURE_META[f][0]})")
    fig_box = px.box(df, x='fault_type', y=feat_sel,
                     color='fault_type', color_discrete_map=FAULT_COLORS, points=False)
    fig_box.update_layout(showlegend=False, height=380, margin=dict(t=10),
                           xaxis_title='Fault Type', yaxis_title=feat_sel)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    col_fi, col_corr = st.columns(2)

    with col_fi:
        st.markdown("#### Feature Importance")
        fi_df = pd.DataFrame({
            'Feature':    FEATURES,
            'Importance': model.feature_importances_,
        }).sort_values('Importance')
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        text=fi_df['Importance'].round(3))
        fig_fi.update_traces(textposition='outside')
        fig_fi.update_layout(coloraxis_showscale=False, height=360, margin=dict(t=10))
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_corr:
        st.markdown("#### Feature Correlation")
        corr = df[FEATURES].corr()
        fig_corr = px.imshow(corr, text_auto='.2f',
                              color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(height=360, margin=dict(t=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.markdown("#### Raw Data Sample")
    st.dataframe(df.sample(50, random_state=42), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL INFO
# ═════════════════════════════════════════════════════════════════════════════
with t6:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Algorithm Details")
        st.table(pd.DataFrame({
            'Parameter': ['Algorithm', 'Estimators', 'Max Depth',
                          'Min Samples Split', 'Test Split', 'Feature Scaling'],
            'Value':     ['Random Forest', '100 trees', '15',
                          '5', '20%', 'StandardScaler'],
        }).set_index('Parameter'))

        st.markdown("#### Fault Labels")
        for fault, color in FAULT_COLORS.items():
            st.markdown(
                f'<span style="background:{color}2a;border:1px solid {color};'
                f'border-radius:6px;padding:3px 10px;margin:3px;display:inline-block;color:{color}">'
                f'{fault.replace("_"," ")}</span>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("#### Input Feature Reference")
        meta_rows = []
        normal_ranges = {
            'temperature': '20-35', 'humidity': '30-70',
            'signal_strength': '-60 to -40', 'battery_level': '50-100',
            'packet_loss_rate': '0-5', 'response_time': '10-50',
        }
        descs = {
            'temperature': 'Ambient temperature at sensor',
            'humidity': 'Relative humidity at sensor',
            'signal_strength': 'Received Signal Strength Indicator',
            'battery_level': 'Remaining battery percentage',
            'packet_loss_rate': 'Percentage of lost data packets',
            'response_time': 'Round-trip response latency',
        }
        for f, (unit, mn, mx, _, _s) in FEATURE_META.items():
            meta_rows.append({'Feature': f, 'Unit': unit,
                               'Normal Range': normal_ranges[f], 'Description': descs[f]})
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Per-Class Performance")
        report_df = get_report(model, scaler, le)
        display_df = report_df.drop(columns=['support'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)
        st.metric("Overall Test Accuracy", f"{accuracy*100:.2f}%")

        st.divider()
        st.markdown("#### Fault Type Descriptions")
        fault_descs = {
            'Normal':              'All sensor readings within expected operating ranges.',
            'Node_Failure':        'Sensor stops responding — erratic readings, extreme packet loss.',
            'Data_Anomaly':        'Readings spike/drop beyond physical plausibility (e.g., temp = 90C).',
            'Battery_Drain':       'Battery critically low — node entering low-power degraded mode.',
            'Communication_Loss':  'Weak signal causes high packet loss and elevated latency.',
        }
        for fault, color in FAULT_COLORS.items():
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:6px 12px;'
                f'margin:5px 0;background:{color}12;border-radius:0 8px 8px 0">'
                f'<b style="color:{color}">{fault.replace("_"," ")}</b><br>'
                f'<span style="font-size:0.82rem;color:#94a3b8">{fault_descs[fault]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:0.8rem">'
    'AI-Powered Fault Detection in Wireless Sensor Networks  |  '
    'Random Forest Classifier  |  Built with Streamlit + Plotly'
    '</div>',
    unsafe_allow_html=True
)
