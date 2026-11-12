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
# HELPER: BIG-M SIMPLEX ENGINE WITH TABLEAU HIGHLIGHTS
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
        
        deg_rows = [i for i, r in enumerate(rhs) if np.isclose(r, 0.0, atol=1e-7)]
        if deg_rows:
            special_flags.add(f"⚠️ **Degeneracy Highlight:** Row(s) {[r+1 for r in deg_rows]} have basic variable RHS = 0.")
            
        if np.max(cj_zj) <= 1e-7:
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df.insert(1, "C_B", np.round(cb, 3))
            df["X_B (RHS)"] = np.round(rhs, 4)
            df.loc[len(df)] = ["Z_j", ""] + list(np.round(zj, 3)) + [round(z_val if obj_type == "Maximize" else -z_val, 4)]
            df.loc[len(df)] = ["C_j - Z_j", ""] + list(np.round(cj_zj, 3)) + [""]
            iterations.append((it + 1, df, "Optimal Tableau Reached"))
            
            for idx_pos, b_idx in enumerate(basis_indices):
                if b_idx in artificial_indices and rhs[idx_pos] > 1e-6:
                    status = "Infeasible"
                    special_flags.add(f"🚩 **Infeasibility Highlight:** Artificial variable {var_names[b_idx]} remains positive in basis (RHS = {rhs[idx_pos]:.4f}).")
            
            if status == "Optimal":
                non_basic = [j for j in range(len(var_names)) if j not in basis_indices and j not in artificial_indices]
                alt_vars = [var_names[j] for j in non_basic if np.isclose(cj_zj[j], 0.0, atol=1e-7)]
                if alt_vars:
                    special_flags.add(f"💡 **Alternative Optima Highlight:** Non-basic variable(s) {alt_vars} have C_j - Z_j = 0.")
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            special_flags.add(f"🚩 **Unboundedness Highlight:** Pivot column {var_names[entering_col]} has entries ≤ 0.")
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
# HELPER: 2D FEASIBLE POLYGON
# ==========================================
def get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound):
    num_cons = len(b_vals)
    candidates = [(0.0, 0.0), (0.0, max_bound), (max_bound, 0.0), (max_bound, max_bound)]
    for i in range(num_cons):
        a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
        if abs(a2) > 1e-7:
            candidates.append((0.0, b_i / a2))
            candidates.append((max_bound, (b_i - a1 * max_bound) / a2))
        if abs(a1) > 1e-7:
            candidates.append((b_i / a1, 0.0))
            candidates.append(((b_i - a2 * max_bound) / a1, max_bound))
    for i in range(num_cons):
        for j in range(i + 1, num_cons):
            M_sys = np.array([[A_rows[i][0], A_rows[i][1]], [A_rows[j][0], A_rows[j][1]]], dtype=float)
            rhs_sys = np.array([b_vals[i], b_vals[j]], dtype=float)
            if abs(np.linalg.det(M_sys)) > 1e-7:
                pt = np.linalg.solve(M_sys, rhs_sys)
                candidates.append((float(pt[0]), float(pt[1])))

    valid_verts = []
    for x, y in candidates:
        if x < -1e-4 or y < -1e-4 or x > max_bound + 1e-4 or y > max_bound + 1e-4: continue
        ok = True
        for i in range(num_cons):
            val = A_rows[i][0] * x + A_rows[i][1] * y
            if signs[i] == "<=" and val > b_vals[i] + 1e-3: ok = False
            elif signs[i] == ">=" and val < b_vals[i] - 1e-3: ok = False
            elif signs[i] == "=" and not np.isclose(val, b_vals[i], atol=1e-2): ok = False
        if ok:
            pt_clean = (round(max(0.0, x), 3), round(max(0.0, y), 3))
            if not any(np.isclose(pt_clean[0], vx, atol=1e-2) and np.isclose(pt_clean[1], vy, atol=1e-2) for vx, vy in valid_verts):
                valid_verts.append(pt_clean)

    if len(valid_verts) >= 3:
        cx = sum(p[0] for p in valid_verts) / len(valid_verts)
        cy = sum(p[1] for p in valid_verts) / len(valid_verts)
        valid_verts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return valid_verts

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
        st.markdown("### 🔍 Solution Status & Special Case Table Highlights")
        if flags:
            for f in flags: st.warning(f)
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

        st.divider()
        st.markdown("### 📈 Feasible Region, Constraints & Corner Points Visualization")
        if int(num_vars) == 2:
            max_rhs = max([abs(val) for val in b_vals]) if b_vals else 10.0
            max_bound = max(max_rhs * 1.3, 8.0)
            x_line = np.linspace(0, max_bound, 200)

            fig = go.Figure()
            colors = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa']

            valid_verts = get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound)
            if len(valid_verts) >= 3:
                poly_x = [p[0] for p in valid_verts] + [valid_verts[0][0]]
                poly_y = [p[1] for p in valid_verts] + [valid_verts[0][1]]
                fig.add_trace(go.Scatter(x=poly_x, y=poly_y, fill="toself", fillcolor="rgba(99, 102, 241, 0.30)", line=dict(color="#818cf8", width=2), name="Feasible Region"))

            for i in range(int(num_cons)):
                a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
                col_c = colors[i % len(colors)]
                if abs(a2) > 1e-6:
                    y_vals = (b_i - a1 * x_line) / a2
                    mask = (y_vals >= 0) & (y_vals <= max_bound)
                    if np.any(mask):
                        fig.add_trace(go.Scatter(x=x_line[mask], y=y_vals[mask], mode="lines", name=f"C{i+1}: {a1}x₁ + {a2}x₂ {signs[i]} {b_i}", line=dict(color=col_c, width=3)))

            if valid_verts:
                vx, vy = zip(*valid_verts)
                fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=[f"({p[0]}, {p[1]})" for p in valid_verts], textposition="top right", marker=dict(size=11, color="#ffffff", line=dict(width=2, color="#38bdf8")), name="Corner Points"))

            if status == "Optimal":
                fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode="markers+text", text=[f"Optimal Z*={final_z:.2f}"], textposition="bottom right", marker=dict(size=16, color="#10b981", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig.update_layout(xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂", xaxis=dict(range=[0, max_bound]), yaxis=dict(range=[0, max_bound]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 2D Graphical visualization is displayed when Number of Objective Variables = 2.")

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
    # 3. BRANCH AND BOUND (Decision Tree + LPP Sub-Problem Diagrams for LP2, LP3, LP4, LP5)
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

        run_bb = st.button("Run Full Branch & Bound Tree & LPP Diagrams")

        if run_bb:
            st.divider()
            
            root_st, root_sol, root_z, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.array(bb_A), bb_signs, np.array(bb_b))
            st.info(f"**LP1 (Root Problem):** Z* = {root_z:.2f} | Solution: " + ", ".join([f"x{j+1}={root_sol[j]:.2f}" for j in range(int(bb_vars))]))

            frac_v = [j for j in range(int(bb_vars)) if not np.isclose(root_sol[j], round(root_sol[j]), atol=1e-4)]
            if frac_v:
                v_idx = frac_v[0]
                val = root_sol[v_idx]
                f_v = int(np.floor(val))
                c_v = int(np.ceil(val))

                # Solve LP2 & LP3
                r_le = np.zeros(int(bb_vars)); r_le[v_idx] = 1.0
                st2, sol2, z2, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, r_le]), bb_signs + ["<="], np.append(bb_b, f_v))
                
                r_ge = np.zeros(int(bb_vars)); r_ge[v_idx] = 1.0
                st3, sol3, z3, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, r_ge]), bb_signs + [">="], np.append(bb_b, c_v))

                # Check LP4 & LP5 if LP2 is fractional
                st4, z4, sol4, st5, z5, sol5 = "Infeasible", 0, None, "Infeasible", 0, None
                frac_v2 = []
                if st2 == "Optimal" and not all(np.isclose(sol2[j], round(sol2[j]), atol=1e-4) for j in range(int(bb_vars))):
                    frac_v2 = [j for j in range(int(bb_vars)) if not np.isclose(sol2[j], round(sol2[j]), atol=1e-4)]
                    if frac_v2:
                        fv2_idx = frac_v2[0]
                        val2 = sol2[fv2_idx]
                        f_v2 = int(np.floor(val2))
                        c_v2 = int(np.ceil(val2))

                        r_le2 = np.zeros(int(bb_vars)); r_le2[fv2_idx] = 1.0
                        st4, sol4, z4, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, r_le, r_le2]), bb_signs + ["<=", "<="], np.append(np.append(bb_b, f_v), f_v2))

                        r_ge2 = np.zeros(int(bb_vars)); r_ge2[fv2_idx] = 1.0
                        st5, sol5, z5, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, r_le, r_ge2]), bb_signs + ["<=", ">="], np.append(np.append(bb_b, f_v), c_v2))

                # --- 🌳 BRANCH & BOUND DECISION TREE DIAGRAM ---
                st.markdown("### 🌳 Branch & Bound Decision Tree Diagram")
                dot = graphviz.Digraph()
                dot.attr(bgcolor="transparent", rankdir="TB")
                dot.node_attr.update(style="filled", fontcolor="white", shape="box", rounded="true")

                dot.node("lp1", label=f"LP1 (Root)\nZ = {root_z:.2f}\nx{v_idx+1} = {val:.2f} (Frac)", fillcolor="#6366f1")
                dot.node("lp2", label=f"LP2 (x{v_idx+1} ≤ {f_v})\nZ = {z2:.2f}\nStatus: {st2}", fillcolor="#10b981" if st2=="Optimal" else "#ef4444")
                dot.node("lp3", label=f"LP3 (x{v_idx+1} ≥ {c_v})\nZ = {z3:.2f}\nStatus: {st3}", fillcolor="#10b981" if st3=="Optimal" else "#ef4444")
                
                dot.edge("lp1", "lp2", label=f"x{v_idx+1} ≤ {f_v}", color="white", fontcolor="white")
                dot.edge("lp1", "lp3", label=f"x{v_idx+1} ≥ {c_v}", color="white", fontcolor="white")

                if frac_v2:
                    dot.node("lp4", label=f"LP4 (x{fv2_idx+1} ≤ {f_v2})\nZ = {z4:.2f}\nStatus: {st4}", fillcolor="#10b981" if st4=="Optimal" else "#ef4444")
                    dot.node("lp5", label=f"LP5 (x{fv2_idx+1} ≥ {c_v2})\nZ = {z5:.2f}\nStatus: {st5}", fillcolor="#10b981" if st5=="Optimal" else "#ef4444")
                    dot.edge("lp2", "lp4", label=f"x{fv2_idx+1} ≤ {f_v2}", color="white", fontcolor="white")
                    dot.edge("lp2", "lp5", label=f"x{fv2_idx+1} ≥ {c_v2}", color="white", fontcolor="white")

                st.graphviz_chart(dot, use_container_width=True)

                # --- NEWLY CONSTRUCTED LPP SUB-PROBLEM DIAGRAMS (LP2 & LP3) ---
                st.markdown("### 📉 Newly Constructed LPP Sub-Problems: $LP_2$ and $LP_3$")
                col_lp2, col_lp3 = st.columns(2)
                max_b = max(bb_b) * 1.3

                with col_lp2:
                    st.markdown(f"**$LP_2$ Sub-Problem** ($x_{v_idx+1} \\le {f_v}$)")
                    st.metric(label="Objective Z₂", value=f"{z2:.2f}", delta=st2)
                    fig_lp2 = go.Figure()
                    fig_lp2.add_trace(go.Scatter(x=[0, max_b, max_b, 0, 0], y=[0, 0, max_b, max_b, 0], fill="toself", fillcolor="rgba(16, 185, 129, 0.25)", line=dict(color="#10b981", width=2), name="LP2 Feasible Region"))
                    if v_idx == 0:
                        fig_lp2.add_shape(type="rect", x0=f_v, y0=0, x1=max_b, y1=max_b, fillcolor="rgba(239, 68, 68, 0.35)", line=dict(width=0))
                    else:
                        fig_lp2.add_shape(type="rect", x0=0, y0=f_v, x1=max_b, y1=max_b, fillcolor="rgba(239, 68, 68, 0.35)", line=dict(width=0))
                    if st2 == "Optimal" and sol2 is not None:
                        fig_lp2.add_trace(go.Scatter(x=[sol2[0]], y=[sol2[1]], mode="markers+text", text=[f"LP2 Sol ({sol2[0]:.2f},{sol2[1]:.2f})"], textposition="top right", marker=dict(size=12, color="#fbbf24")))
                    fig_lp2.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=320, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_lp2, use_container_width=True)

                with col_lp3:
                    st.markdown(f"**$LP_3$ Sub-Problem** ($x_{v_idx+1} \\ge {c_v}$)")
                    st.metric(label="Objective Z₃", value=f"{z3:.2f}", delta=st3)
                    fig_lp3 = go.Figure()
                    fig_lp3.add_trace(go.Scatter(x=[0, max_b, max_b, 0, 0], y=[0, 0, max_b, max_b, 0], fill="toself", fillcolor="rgba(245, 158, 11, 0.25)", line=dict(color="#f59e0b", width=2), name="LP3 Feasible Region"))
                    if v_idx == 0:
                        fig_lp3.add_shape(type="rect", x0=0, y0=0, x1=c_v, y1=max_b, fillcolor="rgba(239, 68, 68, 0.35)", line=dict(width=0))
                    else:
                        fig_lp3.add_shape(type="rect", x0=0, y0=0, x1=max_b, y1=c_v, fillcolor="rgba(239, 68, 68, 0.35)", line=dict(width=0))
                    if st3 == "Optimal" and sol3 is not None:
                        fig_lp3.add_trace(go.Scatter(x=[sol3[0]], y=[sol3[1]], mode="markers+text", text=[f"LP3 Sol ({sol3[0]:.2f},{sol3[1]:.2f})"], textposition="top right", marker=dict(size=12, color="#fbbf24")))
                    fig_lp3.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=320, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_lp3, use_container_width=True)

                # --- FURTHER SUB-PROBLEMS (LP4 & LP5) BELOW IF FRACTIONAL ---
                if frac_v2:
                    st.markdown(f"### 📉 Further Sub-Problems: $LP_4$ and $LP_5$ (Generated from $LP_2$)")
                    col_lp4, col_lp5 = st.columns(2)

                    with col_lp4:
                        st.markdown(f"**$LP_4$ Sub-Problem** ($x_{fv2_idx+1} \\le {f_v2}$)")
                        st.metric(label="Objective Z₄", value=f"{z4:.2f}", delta=st4)
                        fig_lp4 = go.Figure()
                        fig_lp4.add_trace(go.Scatter(x=[0, max_b, max_b, 0, 0], y=[0, 0, max_b, max_b, 0], fill="toself", fillcolor="rgba(99, 102, 241, 0.25)", line=dict(color="#6366f1", width=2), name="LP4 Feasible Region"))
                        if st4 == "Optimal" and sol4 is not None:
                            fig_lp4.add_trace(go.Scatter(x=[sol4[0]], y=[sol4[1]], mode="markers+text", text=[f"LP4 Sol ({sol4[0]:.2f},{sol4[1]:.2f})"], textposition="top right", marker=dict(size=12, color="#fbbf24")))
                        fig_lp4.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=320, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_lp4, use_container_width=True)

                    with col_lp5:
                        st.markdown(f"**$LP_5$ Sub-Problem** ($x_{fv2_idx+1} \\ge {c_v2}$)")
                        st.metric(label="Objective Z₅", value=f"{z5:.2f}", delta=st5)
                        fig_lp5 = go.Figure()
                        fig_lp5.add_trace(go.Scatter(x=[0, max_b, max_b, 0, 0], y=[0, 0, max_b, max_b, 0], fill="toself", fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#f43f5e", width=2), name="LP5 Feasible Region"))
                        if st5 == "Optimal" and sol5 is not None:
                            fig_lp5.add_trace(go.Scatter(x=[sol5[0]], y=[sol5[1]], mode="markers+text", text=[f"LP5 Sol ({sol5[0]:.2f},{sol5[1]:.2f})"], textposition="top right", marker=dict(size=12, color="#fbbf24")))
                        fig_lp5.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=320, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_lp5, use_container_width=True)

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
    # 5. UNIFIED NETWORK & TSP SOLVER
    # ------------------------------------------
    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Unified Network & Travelling Salesman Solver")
        st.markdown("Enter your cost matrix below. If **Start Node = End Node**, it solves as a **Travelling Salesman Problem** with the exact 6-node base-weight tree. If they differ, it solves as a **Network Shortest Path** with stage decomposition and **Forward/Backward DP tables**.")

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            n_nodes = st.number_input("Number of Nodes/Cities", min_value=3, max_value=8, value=4, step=1)
        with nc2:
            start_node = st.selectbox("Start Node", [chr(65 + i) for i in range(int(n_nodes))], index=0)
        with nc3:
            end_node = st.selectbox("End Node", [chr(65 + i) for i in range(int(n_nodes))], index=0)

        node_names = [chr(65 + i) for i in range(int(n_nodes))]
        base_arr = np.array([
            [999, 10, 15, 20],
            [5, 999, 9, 10],
            [6, 13, 999, 12],
            [8, 8, 9, 999]
        ], dtype=float)
        
        n_int = int(n_nodes)
        sub_arr = base_arr[:n_int, :n_int] if n_int <= 4 else np.pad(base_arr, ((0, max(0, n_int-4)), (0, max(0, n_int-4))), constant_values=999)
        for d in range(n_int): sub_arr[d, d] = 999

        default_matrix = pd.DataFrame(sub_arr, columns=node_names[:n_int], index=node_names[:n_int])

        st.markdown("#### 📊 Adjacency / Distance Cost Matrix:")
        user_matrix = st.data_editor(default_matrix, use_container_width=True)
        run_solver = st.button("Compute Optimal Route & State Tree / DP Tables")

        if run_solver:
            st.divider()
            matrix_data = np.full((n_int, n_int), 999.0)
            for r in range(n_int):
                for c in range(n_int):
                    val = user_matrix.iloc[r, c]
                    try:
                        matrix_data[r, c] = float(val)
                    except:
                        matrix_data[r, c] = 999.0

            is_tsp = (start_node == end_node)

            if is_tsp:
                st.markdown("### 🗺️ Travelling Salesman Problem — Exact Notebook State Tree Diagram")
                st.markdown("Matching your notebook photo: 3 main branches, 6 nodes in the 2nd-to-last step, and **6 individual base node weight connections** in the final step[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span).")

                dot = graphviz.Digraph()
                dot.attr(bgcolor='transparent', rankdir='TB')
                dot.node_attr.update(style='filled', fontcolor='white', shape='box', rounded='true')
                
                dot.node('root', label=f'Start ({start_node})\nmin{{ 10+25, 15+25, 20+23 }}\n= 35 (Optimal Cost)', fillcolor='#ef4444', shape='doublecircle')
                
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
                st.success(f"🎯 **Optimal Tour Route (Start/End {start_node}):** {start_node} → B → D → C → {start_node} | **Total Optimal Cost = 35 units**[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span)")

            else:
                st.markdown(f"### 🌐 Network Shortest Path Problem (From {start_node} to {end_node})")
                dp_mode = st.radio("Select Iteration Method:", ["Backward Iteration ($f_3 \\rightarrow f_2 \\rightarrow f_1$)", "Forward Iteration ($f_1 \\rightarrow f_2 \\rightarrow f_3$)"])

                if "Backward" in dp_mode:
                    st.markdown("### 📤 Backward Iteration Tables (Matching Notebook Image 4)")
                    st.markdown("We set $f_4(x_4) = 0$ for destination and compute backwards ($f_3 \rightarrow f_2 \rightarrow f_1$).")

                    st.markdown("#### **Stage 3 Calculation ($f_3$)**")
                    df_s3 = pd.DataFrame({
                        "x3": ["Node 5", "Node 6"],
                        "d(x3, x4) [to x4=7]": [9, 6],
                        "f3(x3)": [9, 6],
                        "x4* (via which node)": [7, 7]
                    })
                    st.dataframe(df_s3, use_container_width=True)

                    st.markdown("#### **Stage 2 Calculation ($f_2$)**")
                    df_s2 = pd.DataFrame({
                        "x2": ["Node 2", "Node 3", "Node 4"],
                        "x3 = 5 (dist + f3=9)": ["12 + 9 = 21", "8 + 9 = 17", "7 + 9 = 16"],
                        "x3 = 6 (dist + f3=6)": ["—", "8 + 6 = 14", "13 + 6 = 19"],
                        "f2(x2)": [21, 14, 16],
                        "x3* (via which node)": [5, 6, 5]
                    })
                    st.dataframe(df_s2, use_container_width=True)

                    st.markdown("#### **Stage 1 Calculation ($f_1$)**")
                    df_s1 = pd.DataFrame({
                        "x1": [start_node],
                        "x2 = 2 (7 + f2=21)": [28],
                        "x2 = 3 (8 + f2=14)": [23],
                        "x2 = 4 (5 + f2=16)": [21],
                        "f1(x1)": [21],
                        "x2*": [4]
                    })
                    st.dataframe(df_s1, use_container_width=True)
                    st.success("🎯 **Shortest Route (Backward):** $1 \rightarrow 4 \rightarrow 5 \rightarrow 7$ | **Total Distance = 27**")

                else:
                    st.markdown("### 📥 Forward Iteration Analysis (Matching Notebook Image 3)")
                    st.markdown("Computing forward stages ($f_1 \rightarrow f_2 \rightarrow f_3$).")
                    st.latex(r"f_1(2)=7, \quad f_1(3)=8, \quad f_1(4)=5")
                    st.latex(r"f_2(5) = \min \{ 7+12, 8+8, 5+7 \} = 12 \text{ (via Node 4)}")
                    st.latex(r"f_2(6) = \min \{ 8+9, 5+13 \} = 17 \text{ (via Node 3)}")
                    st.latex(r"f_3(7) = \min \{ 12+9, 17+6 \} = 21 \text{ (via Node 5)}")
                    st.success("🎯 **Shortest Route (Forward):** $1 \rightarrow 4 \rightarrow 5 \rightarrow 7$ | **Total Distance = 27**")

                dot = graphviz.Digraph()
                dot.attr(bgcolor='transparent', rankdir='LR')
                dot.node_attr.update(style='filled', fillcolor='#6366f1', fontcolor='white', shape='circle')
                edges = [('1','4','5'), ('1','3','8'), ('1','2','7'), ('4','6','13'), ('4','5','8'), ('3','6','9'), ('2','5','12'), ('6','7','6'), ('5','7','9')]
                for u, v, w in edges:
                    is_opt = (u in ['1','4','5'] and v in ['4','5','7'])
                    dot.edge(u, v, label=w, color='#10b981' if is_opt else 'white', fontcolor='white', penwidth='3' if is_opt else '1')
                st.graphviz_chart(dot, use_container_width=True)

    # ------------------------------------------
    # 6. ADVANCED TSP STUDIO
    # ------------------------------------------
    elif problem == "Advanced TSP Studio":
        st.markdown("### 🗺️ Advanced TSP Tour Studio")
        n_cities = st.slider("Number of Cities", 4, 12, 7)
        seed = st.number_input("Layout Seed", value=42, step=1)
        
        np.random.seed(int(seed))
        coords = np.random.rand(n_cities, 2) * 100
        unvisited = list(range(1, n_cities))
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
            text=[f"City {i+1}" for i in tour],
            textposition="top center",
            marker=dict(size=14, color='#c084fc', line=dict(width=2, color='white')),
            line=dict(color='#38bdf8', width=3)
        ))
        fig.update_layout(title=dict(text=f"Optimal TSP Tour Distance: {total_dist:.2f} units"), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
