import streamlit as st
import numpy as np
import plotly.graph_objects as go
import graphviz

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Operations Research Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CLEAN DARK AESTHETIC CSS ---
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none; }
    
    /* Main background & text */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Hero Title */
    .hero-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 3rem;
    }
    
    /* Problem Selection Buttons */
    div.stButton > button {
        width: 100%;
        height: 115px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    div.stButton > button:hover {
        border-color: #818cf8;
        background: rgba(51, 65, 85, 0.9);
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR NAVIGATION ---
if "selected_problem" not in st.session_state:
    st.session_state.selected_problem = None

def select_problem(name):
    st.session_state.selected_problem = name

def go_home():
    st.session_state.selected_problem = None

# ==========================================
# VIEW 1: LANDING SCREEN ("What are we doing today?")
# ==========================================
if st.session_state.selected_problem is None:
    st.markdown('<div class="hero-title">What are we doing today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Select an optimization technique below to enter your parameters and visualize the solution.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.button("📈 Simplex Method", on_click=select_problem, args=("Simplex Method",))
    with col2:
        st.button("🔄 Dual Simplex Method", on_click=select_problem, args=("Dual Simplex Method",))
    with col3:
        st.button("🌳 Branch and Bound", on_click=select_problem, args=("Branch and Bound",))

    st.write("")

    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        st.button("✂️ Gomory Fractional Cut", on_click=select_problem, args=("Gomory Fractional Cut",))
    with col5:
        st.button("🌐 Network Problems", on_click=select_problem, args=("Network Problems",))
    with col6:
        st.button("🗺️ Travelling Salesman Problem", on_click=select_problem, args=("Travelling Salesman Problem",))

# ==========================================
# VIEW 2: DEDICATED PROBLEM WORKSPACE
# ==========================================
else:
    problem = st.session_state.selected_problem
    
    # Top Bar with Back Button
    top_col1, top_col2 = st.columns([1, 6])
    with top_col1:
        st.markdown('<style>div.stButton > button { height: 45px !important; font-size: 1rem !important; }</style>', unsafe_allow_html=True)
        st.button("← Back", on_click=go_home)
    with top_col2:
        st.markdown(f"<h1 style='margin:0; color:#38bdf8;'>{problem}</h1>", unsafe_allow_html=True)
        
    st.divider()

    # 1. SIMPLEX METHOD
    if problem == "Simplex Method":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Objective & Constraints")
            c1 = st.number_input("Objective: x₁ coefficient (c₁)", value=3.0)
            c2 = st.number_input("Objective: x₂ coefficient (c₂)", value=5.0)
            
            st.markdown("**Constraint 1:** $a_{11}x_1 + a_{12}x_2 \\le b_1$")
            a11 = st.number_input("a₁₁", value=1.0)
            a12 = st.number_input("a₁₂", value=2.0)
            b1  = st.number_input("b₁ (RHS)", value=8.0)
            
            st.markdown("**Constraint 2:** $a_{21}x_1 + a_{22}x_2 \\le b_2$")
            a21 = st.number_input("a₂₁", value=1.0)
            a22 = st.number_input("a₂₂", value=1.0)
            b2  = st.number_input("b₂ (RHS)", value=5.0)

        with col2:
            st.subheader("2. Feasible Region & Optimal Contour")
            x1_vals = np.linspace(0, 12, 200)
            x2_vals = np.linspace(0, 12, 200)
            X1, X2 = np.meshgrid(x1_vals, x2_vals)
            
            Z_space = c1 * X1 + c2 * X2
            feasible = (a11 * X1 + a12 * X2 <= b1) & (a21 * X1 + a22 * X2 <= b2) & (X1 >= 0) & (X2 >= 0)
            Z_space[~feasible] = np.nan

            fig = go.Figure()
            if a12 != 0:
                fig.add_trace(go.Scatter(x=x1_vals, y=(b1 - a11*x1_vals)/a12, mode='lines', name='Constraint 1', line=dict(color='#38bdf8', width=3)))
            if a22 != 0:
                fig.add_trace(go.Scatter(x=x1_vals, y=(b2 - a21*x1_vals)/a22, mode='lines', name='Constraint 2', line=dict(color='#f472b6', width=3)))
            fig.add_trace(go.Contour(x=x1_vals, y=x2_vals, z=Z_space, colorscale='Plasma', showscale=True, opacity=0.85))
            fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", yaxis_range=[0, 10], xaxis_range=[0, 10], template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)')
            st.plotly_chart(fig, use_container_width=True)

    # 2. DUAL SIMPLEX METHOD
    elif problem == "Dual Simplex Method":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Infeasible RHS Values")
            rhs1 = st.number_input("Constraint 1 RHS (negative)", value=-4.0)
            rhs2 = st.number_input("Constraint 2 RHS (negative)", value=-6.0)
            step = st.button("Run Pivot Step")
        with col2:
            st.subheader("2. Dual Simplex Tableau")
            tableau = np.array([
                [-3.0, -4.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 1.0, 0.0, rhs1],
                [-2.0, -1.0, 0.0, 1.0, rhs2]
            ])
            st.dataframe(tableau, use_container_width=True)
            if step:
                leaving = "Row 1" if rhs1 < rhs2 else "Row 2"
                st.success(f"Leaving Variable selected: **{leaving}** (most negative RHS). Ratio test determines entering variable.")

    # 3. BRANCH AND BOUND
    elif problem == "Branch and Bound":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Integer Problem Parameters")
            root_z = st.number_input("LP Relaxation Objective (Z)", value=45.2)
            frac_x = st.number_input("Fractional Variable Value (x₁)", value=2.4)
            st.info(f"Branching on x₁ ≤ {int(np.floor(frac_x))} and x₁ ≥ {int(np.ceil(frac_x))}")
        with col2:
            st.subheader("2. Decision Tree Visualization")
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent', rankdir='TB')
            dot.node_attr.update(style='filled', fontcolor='white', shape='box')
            dot.node('0', f'Root LP\nZ = {root_z}\nx₁ = {frac_x}', fillcolor='#6366f1')
            dot.node('1', f'Node 1 (x₁ ≤ {int(np.floor(frac_x))})\nZ = {round(root_z - 3.2, 1)}', fillcolor='#10b981')
            dot.node('2', f'Node 2 (x₁ ≥ {int(np.ceil(frac_x))})\nInfeasible (Pruned)', fillcolor='#ef4444')
            dot.node('3', f'Optimal Integer\nZ = {round(root_z - 5.2, 1)}', fillcolor='#f59e0b')
            dot.edge('0', '1', label=f'x₁ ≤ {int(np.floor(frac_x))}', color='white', fontcolor='white')
            dot.edge('0', '2', label=f'x₁ ≥ {int(np.ceil(frac_x))}', color='white', fontcolor='white')
            dot.edge('1', '3', label='Integer Found', color='white', fontcolor='white')
            st.graphviz_chart(dot, use_container_width=True)

    # 4. GOMORY FRACTIONAL CUT
    elif problem == "Gomory Fractional Cut":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Cut Parameters")
            cut_offset = st.slider("Cutting Plane Tightness", 2.5, 4.0, 3.3, 0.1)
        with col2:
            st.subheader("2. Cutting Plane Visualization")
            x1 = np.linspace(0, 5, 100)
            y1 = 4.2 - 0.8 * x1
            cut = cut_offset - 0.7 * x1
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Original LP Boundary', line=dict(color='#38bdf8', width=3)))
            fig.add_trace(go.Scatter(x=x1, y=cut, mode='lines', name='Gomory Fractional Cut', line=dict(color='#f43f5e', width=3, dash='dash')))
            fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", yaxis_range=[0, 5], template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # 5. NETWORK PROBLEMS
    elif problem == "Network Problems":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Configure Network")
            net_type = st.selectbox("Algorithm", ["Dijkstra (Shortest Path)", "Ford-Fulkerson (Max Flow)"])
            w_ab = st.slider("Edge A → B Weight/Capacity", 1, 20, 4)
            w_bc = st.slider("Edge B → C Weight/Capacity", 1, 20, 2)
            w_ac = st.slider("Edge A → C Weight/Capacity", 1, 20, 7)
        with col2:
            st.subheader("2. Network Graph")
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent', rankdir='LR')
            dot.node_attr.update(style='filled', fillcolor='#6366f1', fontcolor='white', shape='circle')
            
            if net_type == "Dijkstra (Shortest Path)":
                via_b = (w_ab + w_bc) <= w_ac
                dot.edge('A', 'B', label=str(w_ab), color='#10b981' if via_b else 'white', fontcolor='white', penwidth='3' if via_b else '1')
                dot.edge('B', 'C', label=str(w_bc), color='#10b981' if via_b else 'white', fontcolor='white', penwidth='3' if via_b else '1')
                dot.edge('A', 'C', label=str(w_ac), color='#10b981' if not via_b else 'white', fontcolor='white', penwidth='3' if not via_b else '1')
                st.graphviz_chart(dot, use_container_width=True)
                st.success(f"Shortest Path Cost from A to C: **{min(w_ab + w_bc, w_ac)}**")
            else:
                max_flow = min(w_ab, w_bc) + w_ac
                dot.edge('A', 'B', label=f"{min(w_ab, w_bc)}/{w_ab}", color='#38bdf8', fontcolor='white', penwidth='2')
                dot.edge('B', 'C', label=f"{min(w_ab, w_bc)}/{w_bc}", color='#38bdf8', fontcolor='white', penwidth='2')
                dot.edge('A', 'C', label=f"{w_ac}/{w_ac}", color='#c084fc', fontcolor='white', penwidth='2')
                st.graphviz_chart(dot, use_container_width=True)
                st.success(f"Maximum Flow from A to C: **{max_flow}**")

    # 6. TRAVELLING SALESMAN PROBLEM
    elif problem == "Travelling Salesman Problem":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Configure Cities")
            num_cities = st.slider("Number of Cities", 4, 15, 7)
            seed = st.number_input("Map Layout Seed", value=42, step=1)
        with col2:
            st.subheader("2. Tour Visualization")
            np.random.seed(int(seed))
            coords = np.random.rand(num_cities, 2) * 100
            
            unvisited = list(range(1, num_cities))
            tour = [0]
            while unvisited:
                last = tour[-1]
                next_city = min(unvisited, key=lambda c: np.linalg.norm(coords[last] - coords[c]))
                tour.append(next_city)
                unvisited.remove(next_city)
            tour.append(0)
            
            total_dist = sum(np.linalg.norm(coords[tour[i]] - coords[tour[i+1]]) for i in range(len(tour)-1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=coords[tour, 0], y=coords[tour, 1],
                mode='lines+markers+text',
                text=[f"City {i}" for i in tour],
                textposition="top center",
                marker=dict(size=14, color='#c084fc', line=dict(width=2, color='white')),
                line=dict(color='#38bdf8', width=3)
            ))
            fig.update_layout(title=f"Total Tour Distance: {total_dist:.2f} units", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
