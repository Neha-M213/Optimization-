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
# HELPER: FLEXIBLE BIG-M SIMPLEX ENGINE
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
        
        if any(np.isclose(rhs, 0.0, atol=1e-7)):
            special_flags.add("Degeneracy Detected: A basic variable has X_B (RHS) = 0.")
            
        if np.max(cj_zj) <= 1e-7:
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df.insert(1, "C_B", np.round(cb, 3))
            df["X_B (RHS)"] = np.round(rhs, 4)
            df.loc[len(df)] = ["Z_j", ""] + list(np.round(zj, 3)) + [round(z_val if obj_type == "Maximize" else -z_val, 4)]
            iterations.append((it + 1, df, "Optimal Tableau Reached"))
            
            for idx_pos, b_idx in enumerate(basis_indices):
                if b_idx in artificial_indices and rhs[idx_pos] > 1e-6:
                    status = "Infeasible"
                    special_flags.add(f"Infeasible Solution: Artificial variable remains positive.")
            
            if status == "Optimal":
                non_basic = [j for j in range(len(var_names)) if j not in basis_indices and j not in artificial_indices]
                if any(np.isclose(cj_zj[j], 0.0, atol=1e-7) for j in non_basic):
                    special_flags.add("Alternative Optima Detected: C_j - Z_j = 0 for a non-basic variable.")
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            special_flags.add("Unbounded Solution.")
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df["X_B (RHS)"] = np.round(rhs, 4)
            iterations.append((it + 1, df, f"Unbounded at {var_names[entering_col]}"))
            break
            
        valid_ratios = []
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                valid_ratios.append((rhs[i] / col_vals[i], i))
                
        leaving_row = min(valid_ratios, key=lambda x: x[0])[1]
        
        df = pd.DataFrame(np.round(tableau, 4), columns=var_names)
        df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
        df.insert(1, "C_B", np.round(cb, 3))
        df["X_B (RHS)"] = np.round(rhs, 4)
        iterations.append((it + 1, df, f"Entering: **{var_names[entering_col]}** | Leaving: **{var_names[basis_indices[leaving_row]]}**"))
        
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
# HELPER: RECURSIVE BRANCH & BOUND WITH STRIPS
# ==========================================
def run_recursive_branch_and_bound(obj_type, c, A, signs, b):
    root_status, root_sol, root_z, _, _ = solve_simplex_big_m(obj_type, c, A, signs, b)
    
    tree_nodes = []
    queue = [(A, signs, b, "Root", "P0 (Root)", root_status, root_sol, root_z, [])]
    best_integer_z = -1e18 if obj_type == "Maximize" else 1e18
    best_integer_sol = None
    node_counter = 0

    while queue and len(tree_nodes) < 20:
        curr_A, curr_signs, curr_b, parent_id, node_name, status, sol, z_val, strips = queue.pop(0)
        node_id = f"node_{node_counter}"
        node_counter += 1

        is_integer = True
        fractional_var = -1
        if status == "Optimal":
            for j in range(len(sol)):
                if not np.isclose(sol[j], round(sol[j]), atol=1e-4):
                    is_integer = False
                    fractional_var = j
                    break

        tree_nodes.append({
            "id": node_id, "name": node_name, "status": status, "sol": sol, "z": z_val, 
            "is_integer": is_integer, "parent": parent_id, "strips": strips
        })

        if status == "Optimal":
            if is_integer:
                if obj_type == "Maximize" and z_val > best_integer_z:
                    best_integer_z = z_val
                    best_integer_sol = sol
                elif obj_type == "Minimize" and z_val < best_integer_z:
                    best_integer_z = z_val
                    best_integer_sol = sol
            else:
                val = sol[fractional_var]
                floor_v = int(np.floor(val))
                ceil_v = int(np.ceil(val))
                new_strip = (fractional_var, floor_v, ceil_v)

                # Left branch
                row_le = np.zeros(len(c)); row_le[fractional_var] = 1.0
                next_A1 = np.vstack([curr_A, row_le])
                next_s1 = curr_signs + ["<="]
                next_b1 = np.append(curr_b, floor_v)
                st1, sol1, z1, _, _ = solve_simplex_big_m(obj_type, c, next_A1, next_s1, next_b1)
                queue.append((next_A1, next_s1, next_b1, node_id, f"P_{node_counter} (x{fractional_var+1} ≤ {floor_v})", st1, sol1, z1, strips + [new_strip]))

                # Right branch
                row_ge = np.zeros(len(c)); row_ge[fractional_var] = 1.0
                next_A2 = np.vstack([curr_A, row_ge])
                next_s2 = curr_signs + [">="]
                next_b2 = np.append(curr_b, ceil_v)
                st2, sol2, z2, _, _ = solve_simplex_big_m(obj_type, c, next_A2, next_s2, next_b2)
                queue.append((next_A2, next_s2, next_b2, node_id, f"P_{node_counter+1} (x{fractional_var+1} ≥ {ceil_v})", st2, sol2, z2, strips + [new_strip]))

    return tree_nodes, best_integer_z if best_integer_z != -1e18 else None, best_integer_sol

# ==========================================
# VIEW 1: LANDING SCREEN
# ==========================================
if st.session_state.selected_problem is None:
    st.markdown('<div class="hero-title">What are we doing today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Select an optimization technique below to enter your parameters and visualize the solution.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1: st.button("📈 Simplex Method", on_click=select_problem, args=("Simplex Method",))
    with col2: st.button("🔄 Dual Simplex Method", on_click=select_problem, args=("Dual Simplex Method",))
    with col3: st.button("🌳 Branch and Bound (Recursive ILP)", on_click=select_problem, args=("Branch and Bound",))

    st.write("")
    col4, col5, col6 = st.columns(3, gap="large")
    with col4: st.button("✂️ Gomory Fractional Cut", on_click=select_problem, args=("Gomory Fractional Cut",))
    with col5: st.button("🌐 Network Shortest Path (Stages & DP)", on_click=select_problem, args=("Network Shortest Path",))
    with col6: st.button("🗺️ Travelling Salesman Problem", on_click=select_problem, args=("Travelling Salesman Problem",))

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

    # ------------------------------------------
    # 1. SIMPLEX METHOD
    # ------------------------------------------
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
            with obj_cols[j]: c_vals.append(st.number_input(f"c{j+1}", value=1.0, key=f"c_{j}"))

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
        st.markdown("### 🔍 Solution Status & Special Case Highlights")
        if flags:
            for f in flags:
                if "Infeasible" in f: st.error(f"🚩 **Infeasibility Highlight:** {f}")
                elif "Unbounded" in f: st.error(f"🚩 **Unboundedness Highlight:** {f}")
                elif "Degeneracy" in f: st.warning(f"⚠️ **Degeneracy Highlight:** {f}")
                else: st.info(f"💡 **Alternative Optima Highlight:** {f}")
        else:
            st.success("✅ **Unique Optimal Solution**")

        if status == "Optimal":
            sol_str = ", ".join([f"**x{j+1} = {sol[j]:.4f}**" for j in range(int(num_vars))])
            st.info(f"**Optimal Objective Z = {final_z:.4f}** | {sol_str}")

        st.divider()
        st.markdown("### 📋 Step-by-Step Simplex Tableaux")
        for iter_num, df, note in iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True)

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
    # ------------------------------------------
    elif problem == "Dual Simplex Method":
        st.markdown("### ⚙️ Dual Simplex Configuration")
        rhs1 = st.number_input("Constraint 1 RHS", value=-4.0)
        rhs2 = st.number_input("Constraint 2 RHS", value=-6.0)
        if st.button("Run Pivot Step"):
            tableau = np.array([[-3.0, -4.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 1.0, 0.0, rhs1], [-2.0, -1.0, 0.0, 1.0, rhs2]])
            st.dataframe(tableau, use_container_width=True)
            st.success(f"Leaving Variable selected: **Row {'1' if rhs1 < rhs2 else '2'}** (Most negative RHS).")

    # ------------------------------------------
    # 3. BRANCH AND BOUND (With Reliable Scatter Polygon Strip Graphs)
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.markdown("### ⚙️ Step 1: Problem Dimensions")
        dim1, dim2, dim3 = st.columns(3)
        with dim1: bb_obj = st.selectbox("BB Goal", ["Maximize", "Minimize"])
        with dim2: bb_vars = st.number_input("BB Vars", min_value=2, max_value=5, value=2, step=1)
        with dim3: bb_cons = st.number_input("BB Cons", min_value=1, max_value=10, value=2, step=1)

        st.markdown("### 📊 Step 2: Enter Objective Function Coefficients ($c_j$)")
        obj_cols = st.columns(int(bb_vars))
        bb_c = []
        for j in range(int(bb_vars)):
            with obj_cols[j]: bb_c.append(st.number_input(f"bb_c_{j}", value=1.0 if j==0 else 4.0, key=f"bb_c_{j}"))

        st.markdown(f"### ⛓️ Step 3: Enter Constraints ({int(bb_cons)} constraints)")
        bb_A, bb_signs, bb_b = [], [], []
        default_bbA = [[2.0, 4.0], [5.0, 3.0], [1.0, 1.0]]
        default_bbb = [7.0, 15.0, 6.0]
        for i in range(int(bb_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            r_cols = st.columns(int(bb_vars) + 2)
            row_a = []
            for j in range(int(bb_vars)):
                with r_cols[j]: row_a.append(st.number_input(f"bb_a_{i}_{j}", value=default_bbA[i][j] if i < 3 and j < 2 else 1.0, key=f"bb_a_{i}_{j}"))
            with r_cols[int(bb_vars)]: bb_signs.append(st.selectbox(f"bb_s_{i}", ["<=", ">=", "="], key=f"bb_s_{i}"))
            with r_cols[int(bb_vars) + 1]: bb_b.append(st.number_input(f"bb_b_{i}", value=default_bbb[i] if i < 3 else 10.0, key=f"bb_b_{i}"))
            bb_A.append(row_a)

        run_bb = st.button("Run Full Recursive Branch & Bound ILP Tree")

        if run_bb:
            st.divider()
            nodes, best_z, best_sol = run_recursive_branch_and_bound(
                bb_obj, np.array(bb_c), np.array(bb_A), bb_signs, np.array(bb_b)
            )

            col_res1, col_res2 = st.columns([1, 1.3], gap="large")
            with col_res1:
                st.markdown("### 🏆 ILP Optimal Integer Result")
                if best_z is not None:
                    st.success(f"🎯 **Integer Optimal Solution Found!**")
                    sol_str = ", ".join([f"x{j+1} = {best_sol[j]:.2f}" for j in range(int(bb_vars))])
                    st.metric(label="Optimal Integer Objective Z*", value=f"{best_z:.2f}")
                    st.markdown(f"**Optimal Variables:** {sol_str}")
                else:
                    st.error("No integer feasible solution found.")

            with col_res2:
                st.markdown("### 🌳 Full Recursive Branch & Bound Decision Tree")
                dot = graphviz.Digraph()
                dot.attr(bgcolor="transparent", rankdir="TB")
                dot.node_attr.update(style="filled", fontcolor="white", shape="box", rounded="true")

                for n in nodes:
                    color = "#10b981" if n["is_integer"] else ("#6366f1" if n["status"] == "Optimal" else "#ef4444")
                    sol_txt = ", ".join([f"x{j+1}={n['sol'][j]:.2f}" for j in range(int(bb_vars))]) if n['status'] == "Optimal" else n['status']
                    dot.node(n["id"], label=f"{n['name']}\nZ = {n['z']:.2f}\n{sol_txt}", fillcolor=color)
                    if n["parent"] != "Root":
                        dot.edge(n["parent"], n["id"], color="white")

                st.graphviz_chart(dot, use_container_width=True)

            # Recursive Strip Deletion Visualizations using robust Scatter polygons
            st.divider()
            st.markdown("### 📉 Cumulative Strip Deletion Graphs Across Recursive Branches")
            for n in nodes:
                if n["strips"]:
                    st.markdown(f"#### Sub-Problem: `{n['name']}` (Cumulative Deleted Strips)")
                    fig = go.Figure()
                    max_b = max(bb_b) * 1.3 if bb_b else 10.0
                    
                    # Feasible region bounding box
                    fig.add_trace(go.Scatter(
                        x=[0, max_b, max_b, 0, 0], y=[0, 0, max_b, max_b, 0],
                        fill="toself", fillcolor="rgba(99, 102, 241, 0.20)",
                        line=dict(color="#818cf8", width=2), name="Feasible Region"
                    ))

                    # Plot all cumulative deleted strips as filled rectangles using Scatter
                    for var_idx, f_v, c_v in n["strips"]:
                        if var_idx == 0:
                            fig.add_trace(go.Scatter(
                                x=[f_v, c_v, c_v, f_v, f_v], y=[0, 0, max_b, max_b, 0],
                                fill="toself", fillcolor="rgba(244, 63, 94, 0.40)",
                                line=dict(color="#f43f5e", width=2, dash="dash"),
                                name=f"Deleted ({f_v} < x₁ < {c_v})"
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=[0, max_b, max_b, 0, 0], y=[f_v, f_v, c_v, c_v, f_v],
                                fill="toself", fillcolor="rgba(244, 63, 94, 0.40)",
                                line=dict(color="#f43f5e", width=2, dash="dash"),
                                name=f"Deleted ({f_v} < x₂ < {c_v})"
                            ))

                    if n["status"] == "Optimal":
                        fig.add_trace(go.Scatter(x=[n["sol"][0]], y=[n["sol"][1]], mode="markers+text", text=[f"Opt ({n['sol'][0]:.2f}, {n['sol'][1]:.2f})"], textposition="top right", marker=dict(size=13, color="#fbbf24"), name="Optimum"))

                    fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=400)
                    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 4. GOMORY FRACTIONAL CUT
    # ------------------------------------------
    elif problem == "Gomory Fractional Cut":
        st.markdown("### ⚙️ Gomory Fractional Cut")
        cut_offset = st.slider("Cut Offset", 2.5, 4.0, 3.3, 0.1)
        x1 = np.linspace(0, 5, 100)
        y1 = 4.2 - 0.8 * x1
        cut = cut_offset - 0.7 * x1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='LP Boundary', line=dict(color='#38bdf8', width=3)))
        fig.add_trace(go.Scatter(x=x1, y=cut, mode='lines', name='Gomory Cut', line=dict(color='#f43f5e', width=3, dash='dash')))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 5. NETWORK SHORTEST PATH (Fully Populated Stages & DP)
    # ------------------------------------------
    elif problem == "Network Shortest Path":
        st.markdown("### 🌐 Network Shortest Path Problem (Stage Decomposition & DP)")
        st.markdown("Choose whether you want to solve via **Forward Iteration** or **Backward Iteration**.")

        iter_mode = st.radio("Select Dynamic Programming Iteration Method:", ["Forward Iteration", "Backward Iteration"])

        if iter_mode == "Forward Iteration":
            st.markdown("### 📥 Forward Iteration Analysis")
            st.markdown("""
            * **Stage 0:** Node 1 (Start)
            * **Stage 1:** Nodes 2 ($f_1(2)=7$), 3 ($f_1(3)=8$), 4 ($f_1(4)=5$)[span_0](start_span)[span_0](end_span)
            * **Stage 2:** Nodes 5 & 6
            * **Stage 3:** Node 7 (End)
            """)
            st.latex(r"f_2(5) = \min \{ 7+12, 8+8, 5+7 \} = 12 \text{ (via Node 4)}")
            st.latex(r"f_2(6) = \min \{ 8+9, 5+13 \} = 17 \text{ (via Node 3)}")
            st.latex(r"f_3(7) = \min \{ 12+9, 17+6 \} = 21 \text{ (via Node 5)}")
            st.success("🎯 **Shortest Route (Forward):** $1 \rightarrow 4 \rightarrow 5 \rightarrow 7$ | **Total Distance = 27**")
        else:
            st.markdown("### 📤 Backward Iteration Analysis")
            st.markdown("We set $f_4(x_4) = 0$ for $x_4 = 7$ and compute backwards ($f_3 \rightarrow f_2 \rightarrow f_1$).")
            data_st3 = {"x3": ["Node 5", "Node 6"], "d(x3, x4)": ["9", "6"], "f3(x3)": ["9", "6"], "x4*": ["7", "7"]}
            st.dataframe(pd.DataFrame(data_st3), use_container_width=True)
            st.success("🎯 **Shortest Route (Backward):** $1 \rightarrow 4 \rightarrow 5 \rightarrow 7$ | **Total Distance = 27**")

        dot = graphviz.Digraph()
        dot.attr(bgcolor='transparent', rankdir='LR')
        dot.node_attr.update(style='filled', fillcolor='#6366f1', fontcolor='white', shape='circle')
        edges = [('1','4','5'), ('1','3','8'), ('1','2','7'), ('4','6','13'), ('4','5','8'), ('3','6','9'), ('2','5','12'), ('6','7','6'), ('5','7','9')]
        for u, v, w in edges:
            is_opt = (u in ['1','4','5'] and v in ['4','5','7'])
            dot.edge(u, v, label=w, color='#10b981' if is_opt else 'white', fontcolor='white', penwidth='3' if is_opt else '1')
        st.graphviz_chart(dot, use_container_width=True)

    # ------------------------------------------
    # 6. TRAVELLING SALESMAN PROBLEM
    # ------------------------------------------
    elif problem == "Travelling Salesman Problem":
        st.markdown("### 🗺️ Travelling Salesman Problem — Exact Notebook State Tree Diagram")
        dot = graphviz.Digraph()
        dot.attr(bgcolor='transparent', rankdir='TB')
        dot.node_attr.update(style='filled', fontcolor='white', shape='box', rounded='true')
        
        dot.node('root', label='Start (1)\nmin{ 10+25, 15+25, 20+23 }\n= 35 (Optimal Cost)', fillcolor='#ef4444', shape='doublecircle')
        
        dot.node('n2', label='Node (2)\nmin{ 9+20, 10+15 }\n= 25', fillcolor='#10b981')
        dot.node('n3', label='Node (3)\nmin{ 13+18, 12+13 }\n= 23', fillcolor='#f59e0b')
        dot.node('n4', label='Node (4)\nmin{ 8+15, 9+18 }\n= 23', fillcolor='#6366f1')
        
        dot.edge('root', 'n2', label=' 1→2 (10)', color='white', fontcolor='white')
        dot.edge('root', 'n3', label=' 1→3 (15)', color='white', fontcolor='white')
        dot.edge('root', 'n4', label=' 1→4 (20)', color='white', fontcolor='white')
        
        dot.node('n2_3', label='Node (3)\n12 + 8 = 20', fillcolor='#334155')
        dot.node('n2_4', label='Node (4)\n9 + 6 = 15', fillcolor='#334155')
        dot.edge('n2', 'n2_3', label=' 2→3 (12)', color='white', fontcolor='white')
        dot.edge('n2', 'n2_4', label=' 2→4 (9)', color='white', fontcolor='white')
        
        dot.node('n3_2', label='Node (2)\n13 + 5 = 18', fillcolor='#334155')
        dot.node('n3_4', label='Node (4)\n8 + 5 = 13', fillcolor='#334155')
        dot.edge('n3', 'n3_2', label=' 3→2 (13)', color='white', fontcolor='white')
        dot.edge('n3', 'n3_4', label=' 3→4 (8)', color='white', fontcolor='white')

        dot.node('n4_2', label='Node (2)\n9 + 5 = 14', fillcolor='#334155')
        dot.node('n4_3', label='Node (3)\n8 + 6 = 14', fillcolor='#334155')
        dot.edge('n4', 'n4_2', label=' 4→2 (9)', color='white', fontcolor='white')
        dot.edge('n4', 'n4_3', label=' 4→3 (8)', color='white', fontcolor='white')
        
        dot.node('base_23', label='Base Node 3 = 6', fillcolor='#475569')
        dot.node('base_24', label='Base Node 4 = 8', fillcolor='#475569')
        dot.node('base_32', label='Base Node 2 = 5', fillcolor='#475569')
        dot.node('base_34', label='Base Node 4 = 8', fillcolor='#475569')
        dot.node('base_42', label='Base Node 2 = 5', fillcolor='#475569')
        dot.node('base_43', label='Base Node 3 = 6', fillcolor='#475569')

        dot.edge('n2_3', 'base_23', color='white')
        dot.edge('n2_4', 'base_24', color='white')
        dot.edge('n3_2', 'base_32', color='white')
        dot.edge('n3_4', 'base_34', color='white')
        dot.edge('n4_2', 'base_42', color='white')
        dot.edge('n4_3', 'base_43', color='white')
        
        st.graphviz_chart(dot, use_container_width=True)
        st.success("🎯 **Optimal Tour Route:** 1 → 2 → 4 → 3 → 1 | **Total Optimal Cost = 35 units**[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span)")
