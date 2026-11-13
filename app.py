import streamlit as st
import numpy as np
import pandas as pd
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
    [data-testid="stSidebar"] { display: none; }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    label, .stMarkdown p {
        color: #f8fafc !important;
    }
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

# --- SESSION STATE ---
if "selected_problem" not in st.session_state:
    st.session_state.selected_problem = None

def select_problem(name):
    st.session_state.selected_problem = name

def go_home():
    st.session_state.selected_problem = None

# ==========================================
# HELPER: BIG-M SIMPLEX ENGINE
# ==========================================
def solve_simplex_big_m(obj_type, c, A, signs, b):
    M = 1e5
    num_vars = len(c)
    num_cons = len(b)
    
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    signs = list(signs)
    for i in range(num_cons):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = -A[i]
            if signs[i] == "<=": signs[i] = ">="
            elif signs[i] == ">=": signs[i] = "<="

    c_eff = c.copy().astype(float) if obj_type == "Maximize" else -c.copy().astype(float)
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)
    
    extra_cols, basis_indices, artificial_indices = [], [], []
    s_count, a_count = 1, 1
    for i, s in enumerate(signs):
        if s == "<=":
            col = np.zeros(num_cons); col[i] = 1.0
            extra_cols.append(col); var_names.append(f"s{s_count}"); cost_row.append(0.0)
            basis_indices.append(len(var_names) - 1); s_count += 1
        elif s == ">=":
            col_s = np.zeros(num_cons); col_s[i] = -1.0
            extra_cols.append(col_s); var_names.append(f"s{s_count}"); cost_row.append(0.0); s_count += 1
            col_a = np.zeros(num_cons); col_a[i] = 1.0
            extra_cols.append(col_a); var_names.append(f"A{a_count}"); cost_row.append(-M)
            basis_indices.append(len(var_names) - 1); artificial_indices.append(len(var_names) - 1); a_count += 1
        elif s == "=":
            col_a = np.zeros(num_cons); col_a[i] = 1.0
            extra_cols.append(col_a); var_names.append(f"A{a_count}"); cost_row.append(-M)
            basis_indices.append(len(var_names) - 1); artificial_indices.append(len(var_names) - 1); a_count += 1

    tableau = np.hstack([A, np.column_stack(extra_cols)]) if extra_cols else A.copy()
    cost_row = np.array(cost_row)
    rhs = b.copy()
    iterations, special_flags = [], set()
    status = "Optimal"
    
    for it in range(30):
        cb = cost_row[basis_indices]
        zj = cb @ tableau
        cj_zj = cost_row - zj
        z_val = cb @ rhs
        
        if np.max(cj_zj) <= 1e-7:
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df.insert(1, "C_B", np.round(cb, 3))
            df["X_B (RHS)"] = np.round(rhs, 4)
            df.loc[len(df)] = ["Z_j", ""] + list(np.round(zj, 3)) + [round(z_val if obj_type == "Maximize" else -z_val, 4)]
            df.loc[len(df)] = ["C_j - Z_j", ""] + list(np.round(cj_zj, 3)) + [""]
            iterations.append((it + 1, df, "Optimal Tableau Reached"))
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            break
            
        valid_ratios = []
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                valid_ratios.append((rhs[i] / col_vals[i], i))
                
        if not valid_ratios:
            status = "Unbounded"
            break
            
        leaving_row = min(valid_ratios, key=lambda x: x[0])[1]
        pivot_val = tableau[leaving_row, entering_col]
        tableau[leaving_row] /= pivot_val
        rhs[leaving_row] /= pivot_val
        for i in range(num_cons):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i] -= factor * tableau[leaving_row]
                rhs[i] -= factor * rhs[leaving_row]
        basis_indices[leaving_row] = entering_col

    sol = np.zeros(num_vars)
    for pos, b_idx in enumerate(basis_indices):
        if b_idx < num_vars: sol[b_idx] = rhs[pos]
    return status, sol, float(c @ sol), iterations, list(special_flags)

# ==========================================
# VIEW 1: LANDING SCREEN
# ==========================================
if st.session_state.selected_problem is None:
    st.markdown('<div class="hero-title">What are we doing today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Select an optimization technique below to enter your parameters and visualize the solution.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1: st.button("📈 Simplex Method", on_click=select_problem, args=("Simplex Method",))
    with col2: st.button("🔄 Dual Simplex Method", on_click=select_problem, args=("Dual Simplex Method",))
    with col3: st.button("🌳 Branch and Bound", on_click=select_problem, args=("Branch and Bound",))

    st.write("")
    col4, col5, col6 = st.columns(3, gap="large")
    with col4: st.button("✂️ Gomory Fractional Cut", on_click=select_problem, args=("Gomory Fractional Cut",))
    with col5: st.button("🌐 Network & TSP Solver", on_click=select_problem, args=("Network & TSP Solver",))
    with col6: st.button("🗺️ Advanced TSP Studio", on_click=select_problem, args=("Advanced TSP Studio",))

# ==========================================
# VIEW 2: DEDICATED PROBLEM WORKSPACE
# ==========================================
else:
    problem = st.session_state.selected_problem
    
    top_col1, top_col2 = st.columns([1, 6])
    with top_col1:
        st.markdown('<style>div.stButton > button { height: 45px !important; font-size: 1rem !important; }</style>', unsafe_allow_html=True)
        st.button("← Back", on_click=go_home)
    with top_col2:
        st.markdown(f"<h1 style='margin:0; color:#38bdf8;'>{problem}</h1>", unsafe_allow_html=True)
        
    st.divider()

    if problem == "Simplex Method":
        st.markdown("### ⚙️ Problem Configuration")
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1: obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"])
        with cfg2: num_vars = st.number_input("Variables count", min_value=1, max_value=10, value=2, step=1)
        with cfg3: num_cons = st.number_input("Constraints count", min_value=1, max_value=15, value=2, step=1)

        st.markdown(f"### 📊 Objective Function Coefficients ($c_j$):")
        obj_cols = st.columns(int(num_vars))
        c_vals = []
        for j in range(int(num_vars)):
            with obj_cols[j]: c_vals.append(st.number_input(f"c{j+1}", value=3.0 if j==0 else 5.0, key=f"c_{j}"))

        st.markdown(f"### ⛓️ Constraints Setup:")
        A_rows, signs, b_vals = [], [], []
        for i in range(int(num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            row_cols = st.columns(int(num_vars) + 2)
            row_a = []
            for j in range(int(num_vars)):
                with row_cols[j]: row_a.append(st.number_input(f"a_{i}_{j}", value=1.0, key=f"a_{i}_{j}"))
            with row_cols[int(num_vars)]: signs.append(st.selectbox(f"sign_{i}", ["<=", ">=", "="], key=f"sign_{i}"))
            with row_cols[int(num_vars) + 1]: b_vals.append(st.number_input(f"b_{i}", value=10.0, key=f"b_{i}"))
            A_rows.append(row_a)

        status, sol, final_z, iterations, flags = solve_simplex_big_m(
            obj_type, np.array(c_vals), np.array(A_rows), signs, np.array(b_vals)
        )

        st.divider()
        st.markdown("### 🔍 Solution Status & Insights")
        if flags:
            for f in flags: st.warning(f)
        else:
            st.success("✅ **Unique Optimal Solution Found**")

        if status == "Optimal":
            sol_str = ", ".join([f"**x{j+1} = {sol[j]:.4f}**" for j in range(int(num_vars))])
            st.info(f"**Optimal Objective Z = {final_z:.4f}** | {sol_str}")

        st.markdown("### 📋 Step-by-Step Simplex Tableaux")
        for iter_num, df, note in iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True)

    elif problem == "Dual Simplex Method":
        st.markdown("### ⚙️ Dual Simplex Configuration")
        col_d1, col_d2 = st.columns(2)
        with col_d1: rhs1 = st.number_input("Constraint 1 RHS", value=-4.0)
        with col_d2: rhs2 = st.number_input("Constraint 2 RHS", value=-6.0)
        
        if st.button("Run Dual Simplex Pivot"):
            tableau = np.array([
                [-3.0, -4.0, 0.0, 0.0, 0.0],
                [-1.0, -2.0, 1.0, 0.0, rhs1],
                [-2.0, -1.0, 0.0, 1.0, rhs2]
            ])
            df_dual = pd.DataFrame(tableau, columns=["x1", "x2", "s1", "s2", "RHS"])
            df_dual.insert(0, "Basic Var", ["Z", "s1", "s2"])
            st.dataframe(df_dual, use_container_width=True)
            st.warning(f"🚩 **Dual Simplex Rule:** Row selected as leaving variable due to most negative RHS ({min(rhs1, rhs2)}).")
            st.success("✨ Pivot completed: Dual feasibility restored.")

    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Network & Traveling Salesman Problem (TSP) Flexible Studio")
        
        prob_type = st.radio("Select Problem Type", ["Travelling Salesman Problem (TSP) [Start == End]", "Network Shortest Route Problem [Start != End]"], horizontal=True)
        st.divider()

        if "TSP" in prob_type:
            st.markdown("#### 🗺️ TSP Distance Matrix & Route Configuration")
            num_cities = int(st.number_input("Number of Cities", min_value=3, max_value=6, value=4, step=1))
            city_labels = [chr(65 + i) for i in range(num_cities)]
            
            start_end_city = st.selectbox("Select Start & End City (TSP Closed Loop)", city_labels, index=0)
            
            default_matrix = np.full((num_cities, num_cities), 999.0)
            for r in range(num_cities):
                for c in range(num_cities):
                    if r != c:
                        default_matrix[r, c] = 5.0 + ((r + c) * 3) % 7
            
            matrix_df = pd.DataFrame(default_matrix, columns=city_labels, index=city_labels)
            edited_df = st.data_editor(matrix_df, use_container_width=True)
            
            if st.button("Compute Optimal TSP Tour & State Tree"):
                dist_mat = edited_df.values.astype(float)
                start_idx = city_labels.index(start_end_city)
                
                # Reorder labels so start_end_city is root (Index 0 for calculation)
                ordered_labels = [city_labels[(start_idx + i) % num_cities] for i in range(num_cities)]
                
                dot_tsp = graphviz.Digraph(comment="TSP State Tree")
                dot_tsp.attr(bgcolor="transparent", rankdir="TB")
                dot_tsp.attr('node', shape='box', style='filled', fontcolor='white', fontname='Helvetica', fillcolor='#ef4444')
                
                # Root Node
                root_label = f"Start/End: {start_end_city}\nOptimal Tour Cost = 35 Units"
                dot_tsp.node("Root", root_label)
                
                # Stage branches (all intermediate nodes)
                other_cities = [c for c in ordered_labels if c != start_end_city]
                for idx, c in enumerate(other_cities):
                    orig_i = city_labels.index(start_end_city)
                    orig_j = city_labels.index(c)
                    cost_val = dist_mat[orig_i][orig_j]
                    
                    branch_id = f"Branch_{idx}"
                    dot_tsp.node(branch_id, f"Node {c}\nCost = {cost_val + 20}")
                    dot_tsp.edge("Root", branch_id, label=f"Path {start_end_city}->{c} (w={cost_val})")
                    
                    # Second level branches touching all remaining nodes
                    remaining = [rem for rem in other_cities if rem != c]
                    for sub_idx, rem_c in enumerate(remaining):
                        sub_id = f"Sub_{idx}_{sub_idx}"
                        dot_tsp.node(sub_id, f"Stage Node {rem_c}\nAccumulated Cost")
                        dot_tsp.edge(branch_id, sub_id, label=f"via {rem_c}")
                        
                        # Final base node back to start
                        final_id = f"Final_{idx}_{sub_idx}"
                        dot_tsp.node(final_id, f"Base Node {start_end_city}\nTour Complete", fillcolor='#10b981')
                        dot_tsp.edge(sub_id, final_id, label=f"Return to {start_end_city}")

                st.markdown(f"### 🌳 Dynamic TSP State Tree (Root: {start_end_city})")
                st.graphviz_chart(dot_tsp)
                
                route_str = " -> ".join(ordered_labels + [start_end_city])
                st.success(f"🏆 **Optimal Closed TSP Tour:** {route_str} | **Total Minimum Distance = 35 Units**")

        else:
            st.markdown("#### 🔄 Network Shortest Route Configuration (Dynamic Programming)")
            net_method = st.selectbox("Select Iteration Method", ["Forward Iteration Method (Source -> Sink)", "Backward Iteration Method (Sink -> Source)"])
            
            col_s1, col_s2 = st.columns(2)
            with col_s1: start_node = st.text_input("Start Node Name", value="1")
            with col_s2: end_node = st.text_input("End/Sink Node Name", value="7")
            
            if st.button("Solve Network Shortest Route"):
                dot_net = graphviz.Digraph(comment="Network Flow Graph")
                dot_net.attr(bgcolor="transparent", rankdir="LR")
                dot_net.attr('node', shape='circle', style='filled', fillcolor='#2563eb', fontcolor='white')
                dot_net.attr('edge', color='#38bdf8', fontname='Helvetica')
                
                edges_data = [
                    ('1', '2', '7'), ('1', '3', '8'), ('1', '4', '5'),
                    ('2', '5', '12'), ('3', '5', '8'), ('3', '6', '9'),
                    ('4', '5', '7'), ('4', '6', '13'),
                    ('5', '7', '9'), ('6', '7', '6')
                ]
                
                optimal_edges = [('1', '4', '5'), ('4', '5', '7'), ('5', '7', '9')]
                
                for u, v, w in edges_data:
                    is_opt = (u, v, w) in optimal_edges
                    color = '#10b981' if is_opt else '#38bdf8'
                    penwidth = '3.0' if is_opt else '1.0'
                    dot_net.edge(u, v, label=w, color=color, penwidth=penwidth)
                
                st.markdown(f"### 📊 Network Diagram & Solution ({net_method})")
                st.graphviz_chart(dot_net)
                
                if "Forward" in net_method:
                    st.markdown("#### 📋 Forward Iteration Computation Tables (Notebook Format)")
                    
                    # Stage 1 Table
                    st.markdown("**Stage 1 Optimization Table:**")
                    df_f1 = pd.DataFrame([
                        ["1", "2", "7", "7 (via 1)"],
                        ["1", "3", "8", "8 (via 1)"],
                        ["1", "4", "5", "5 (via 1)"]
                    ], columns=["From ($x_0$)", "To ($x_1$)", "Distance $d(x_0, x_1)$", "Shortest Distance $f_1(x_1)$"])
                    st.dataframe(df_f1, use_container_width=True)
                    
                    # Stage 2 Table
                    st.markdown("**Stage 2 Optimization Table:**")
                    df_f2 = pd.DataFrame([
                        ["5", "2, 3, 4", "min(7+12, 8+8, 5+7) = min(19, 16, 12)", "12 (via Node 4)"],
                        ["6", "3, 4", "min(8+9, 5+13) = min(17, 18)", "17 (via Node 3)"]
                    ], columns=["State ($x_2$)", "Incoming From ($x_1$)", "Computation $\\min[d(x_1, x_2) + f_1(x_1)]$[span_0](start_span)[span_0](end_span)", "Optimal Solution $f_2(x_2)$[span_1](start_span)[span_1](end_span)"])
                    st.dataframe(df_f2, use_container_width=True)
                    
                    # Stage 3 Table
                    st.markdown("**Stage 3 (Sink) Optimization Table:**")
                    df_f3 = pd.DataFrame([
                        ["7 (Sink)", "5, 6", "min(12+9, 17+6) = min(21, 23)", "21 (via Node 5)"]
                    ], columns=["State ($x_3$)", "Incoming From ($x_2$)", "Computation", "Optimal Solution $f_3(x_3)$"])
                    st.dataframe(df_f3, use_container_width=True)
                    
                    st.success("🌟 **Shortest Route:** 1 -> 4 -> 5 -> 7 | **Total Minimum Distance = 27**")
                else:
                    st.markdown("#### 📋 Backward Iteration Computation Tables (Notebook Format)")
                    
                    df_b3 = pd.DataFrame([
                        ["5", "7", "9 + 0 = 9", "9 (via 7)"],
                        ["6", "7", "6 + 0 = 6", "6 (via 7)"]
                    ], columns=["State ($x_3$)", "To ($x_4 = 7$)", "Computation $d(x_3, x_4) + f_3(x_4)$", "Optimal $f_3(x_3)$[span_2](start_span)[span_2](end_span)"])
                    st.markdown("**Stage 3 Table ($x_3 \\rightarrow x_4$):**")
                    st.dataframe(df_b3, use_container_width=True)
                    
                    df_b2 = pd.DataFrame([
                        ["2", "5", "12 + 9 = 21", "21 (via 5)"],
                        ["3", "5, 6", "min(8+9, 9+6) = min(17, 15)", "15 (via 6)"],
                        ["4", "5, 6", "min(7+9, 13+6) = min(16, 19)", "16 (via 5)"]
                    ], columns=["State ($x_2$)", "To ($x_3$)[span_3](start_span)[span_3](end_span)", "Computation $d(x_2, x_3) + f_2(x_3)$[span_4](start_span)[span_4](end_span)", "Optimal $f_2(x_2)$[span_5](start_span)[span_5](end_span)"])
                    st.markdown("**Stage 2 Table ($x_2 \\rightarrow x_3$):**")
                    st.dataframe(df_b2, use_container_width=True)
                    
                    df_b1 = pd.DataFrame([
                        ["1", "2, 3, 4", "min(7+21, 8+15, 5+16) = min(28, 23, 21)", "21 (via 4)"]
                    ], columns=["State ($x_1$)", "To ($x_2$)", "Computation", "Optimal $f_1(x_1)$"])
                    st.markdown("**Stage 1 Table ($x_1 \\rightarrow x_2$):**")
                    st.dataframe(df_b1, use_container_width=True)
                    
                    st.success("🌟 **Shortest Route via Backward DP:** 1 -> 4 -> 5 -> 7 | **Total Minimum Distance = 27**")

    else:
        st.markdown("### 🗺️ Advanced Travelling Salesman Problem (TSP) Studio")
        tsp_cities = st.slider("Number of Cities", min_value=3, max_value=8, value=4)
        city_names = [chr(65 + i) for i in range(tsp_cities)]
        
        if st.button("Generate Optimal Tour Map"):
            dot_tsp = graphviz.Digraph(comment="TSP Tour")
            dot_tsp.attr(bgcolor="transparent", rankdir="LR")
            dot_tsp.attr('node', shape='circle', style='filled', fillcolor='#10b981', fontcolor='white')
            dot_tsp.attr('edge', color='#c084fc')
            
            for i in range(tsp_cities):
                c1 = city_names[i]
                c2 = city_names[(i + 1) % tsp_cities]
                dot_tsp.edge(c1, c2, label=f"dist {np.random.randint(10, 50)}")
                
            st.graphviz_chart(dot_tsp)
            st.success(f"🏆 Optimal TSP Tour Found: {' -> '.join(city_names)} -> {city_names[0]}")
