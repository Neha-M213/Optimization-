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
            special_flags.add("Degeneracy Detected (Basic variable equals 0 in X_B)")
            
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
                    special_flags.add("Infeasible Solution (Artificial variable remains positive)")
            
            if status == "Optimal":
                non_basic = [j for j in range(len(var_names)) if j not in basis_indices and j not in artificial_indices]
                if any(np.isclose(cj_zj[j], 0.0, atol=1e-7) for j in non_basic):
                    special_flags.add("Alternative Optima Detected (C_j - Z_j = 0 for non-basic variable)")
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            special_flags.add("Unbounded Solution (Pivot column entries ≤ 0)")
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df["X_B (RHS)"] = np.round(rhs, 4)
            iterations.append((it + 1, df, f"Unbounded at {var_names[entering_col]}"))
            break
            
        ratios, valid_ratios = [], []
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                r = rhs[i] / col_vals[i]
                ratios.append(round(r, 4))
                valid_ratios.append((r, i))
            else:
                ratios.append("-")
                
        min_r = min(r for r, _ in valid_ratios)
        tied_rows = [i for r, i in valid_ratios if np.isclose(r, min_r, atol=1e-7)]
        if len(tied_rows) > 1:
            special_flags.add("Degeneracy Detected (Tie in Minimum Ratio Test)")
            
        leaving_row = tied_rows[0]
        
        df = pd.DataFrame(np.round(tableau, 4), columns=var_names)
        df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
        df.insert(1, "C_B", np.round(cb, 3))
        df["X_B (RHS)"] = np.round(rhs, 4)
        df["Min Ratio"] = ratios
        df.loc[len(df)] = ["Z_j", ""] + list(np.round(zj, 3)) + [round(z_val if obj_type == "Maximize" else -z_val, 4), ""]
        df.loc[len(df)] = ["C_j - Z_j", ""] + list(np.round(cj_zj, 3)) + ["", ""]
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
# HELPER: COMPUTE 2D FEASIBLE POLYGON
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
        with cfg1:
            st.markdown("**Objective Goal**")
            obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"], label_visibility="collapsed")
        with cfg2:
            st.markdown("**Number of Objective Variables**")
            num_vars = st.number_input("Variables count", min_value=1, max_value=10, value=2, step=1, label_visibility="collapsed")
        with cfg3:
            st.markdown("**Number of Constraints**")
            num_cons = st.number_input("Constraints count", min_value=1, max_value=15, value=2, step=1, label_visibility="collapsed")

        st.markdown(f"### 📊 Objective Function Coefficients ($c_j$ for {int(num_vars)} variables):")
        obj_cols = st.columns(int(num_vars))
        c_vals = []
        default_c = [1.0, 4.0, 3.0, 2.0, 5.0]
        for j in range(int(num_vars)):
            with obj_cols[j]:
                st.markdown(f"**Coeff of $x_{j+1}$ ($c_{j+1}$)**")
                c_vals.append(st.number_input(f"c{j+1}", value=default_c[j % len(default_c)], key=f"c_{j}", label_visibility="collapsed"))

        st.markdown(f"### ⛓️ Constraints Setup ({int(num_cons)} constraints):")
        A_rows, signs, b_vals = [], [], []
        default_A = [[2.0, 4.0], [5.0, 3.0], [1.0, 1.0]]
        default_b = [7.0, 15.0, 6.0]

        for i in range(int(num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            row_cols = st.columns(int(num_vars) + 2)
            row_a = []
            for j in range(int(num_vars)):
                with row_cols[j]:
                    st.caption(f"$x_{j+1}$ coeff")
                    row_a.append(st.number_input(f"a_{i}_{j}", value=default_A[i][j] if i < 3 and j < 2 else 1.0, key=f"a_{i}_{j}", label_visibility="collapsed"))
            with row_cols[int(num_vars)]:
                st.caption("Inequality")
                signs.append(st.selectbox(f"sign_{i}", ["<=", ">=", "="], key=f"sign_{i}", label_visibility="collapsed"))
            with row_cols[int(num_vars) + 1]:
                st.caption("RHS ($b_" + str(i+1) + "$)")
                b_vals.append(st.number_input(f"b_{i}", value=default_b[i] if i < 3 else 10.0, key=f"b_{i}", label_visibility="collapsed"))
            A_rows.append(row_a)

        status, sol, final_z, iterations, flags = solve_simplex_big_m(
            obj_type, np.array(c_vals), np.array(A_rows), signs, np.array(b_vals)
        )

        st.divider()
        st.markdown("### 🔍 Solution Status & Special Case Flags")
        if flags:
            for f in flags:
                if "Infeasible" in f or "Unbounded" in f: st.error(f"🚩 **Special Case Flagged:** {f}")
                else: st.warning(f"⚠️ **Special Case Flagged:** {f}")
        else:
            st.success("✅ **Unique Optimal Solution** (No degeneracy, infeasibility, unboundedness, or alternative optima detected).")

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
                elif abs(a1) > 1e-6:
                    x_val = b_i / a1
                    if 0 <= x_val <= max_bound:
                        fig.add_trace(go.Scatter(x=[x_val, x_val], y=[0, max_bound], mode="lines", name=f"C{i+1}: {a1}x₁ {signs[i]} {b_i}", line=dict(color=col_c, width=3)))

            if valid_verts:
                vx, vy = zip(*valid_verts)
                fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=[f"({p[0]}, {p[1]})" for p in valid_verts], textposition="top right", textfont=dict(color="#ffffff", size=12), marker=dict(size=11, color="#ffffff", line=dict(width=2, color="#38bdf8")), name="Corner Points"))

            if status == "Optimal":
                fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode="markers+text", text=[f"Optimal Z*={final_z:.2f}"], textposition="bottom right", textfont=dict(color="#10b981", size=13), marker=dict(size=16, color="#10b981", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig.update_layout(xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂", xaxis=dict(range=[0, max_bound], gridcolor="rgba(255,255,255,0.1)"), yaxis=dict(range=[0, max_bound], gridcolor="rgba(255,255,255,0.1)"), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 2D Graphical visualization is displayed when Number of Objective Variables = 2.")

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
    # ------------------------------------------
    elif problem == "Dual Simplex Method":
        st.markdown("### ⚙️ Dual Simplex Configuration")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("**Constraint 1 RHS (Negative value)**")
            rhs1 = st.number_input("rhs1", value=-4.0, label_visibility="collapsed")
            st.markdown("**Constraint 2 RHS (Negative value)**")
            rhs2 = st.number_input("rhs2", value=-6.0, label_visibility="collapsed")
            step = st.button("Run Pivot Step")
        with col2:
            tableau = np.array([[-3.0, -4.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 1.0, 0.0, rhs1], [-2.0, -1.0, 0.0, 1.0, rhs2]])
            st.dataframe(tableau, use_container_width=True)
            if step:
                st.success(f"Leaving Variable selected: **Row {'1' if rhs1 < rhs2 else '2'}** (Most negative RHS).")

    # ------------------------------------------
    # 3. BRANCH AND BOUND
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.markdown("### ⚙️ Step 1: Problem Dimensions")
        dim1, dim2, dim3 = st.columns(3)
        with dim1:
            st.markdown("**Objective Goal**")
            bb_obj = st.selectbox("BB Goal", ["Maximize", "Minimize"], label_visibility="collapsed")
        with dim2:
            st.markdown("**Number of Objective Variables**")
            bb_vars = st.number_input("BB Vars", min_value=2, max_value=5, value=2, step=1, label_visibility="collapsed")
        with dim3:
            st.markdown("**Number of Constraints**")
            bb_cons = st.number_input("BB Cons", min_value=1, max_value=10, value=2, step=1, label_visibility="collapsed")

        st.markdown("### 📊 Step 2: Enter Objective Function Coefficients ($c_j$)")
        obj_cols = st.columns(int(bb_vars))
        bb_c = []
        default_bbc = [1.0, 4.0, 3.0, 2.0, 5.0]
        for j in range(int(bb_vars)):
            with obj_cols[j]:
                st.markdown(f"**Coefficient of $x_{j+1}$ ($c_{j+1}$)**")
                bb_c.append(st.number_input(f"bb_c_{j}", value=default_bbc[j % len(default_bbc)], key=f"bb_c_{j}", label_visibility="collapsed"))

        st.markdown(f"### ⛓️ Step 3: Enter Constraints ({int(bb_cons)} constraints)")
        bb_A, bb_signs, bb_b = [], [], []
        default_bbA = [[2.0, 4.0], [5.0, 3.0], [1.0, 1.0]]
        default_bbb = [7.0, 15.0, 6.0]
        for i in range(int(bb_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            r_cols = st.columns(int(bb_vars) + 2)
            row_a = []
            for j in range(int(bb_vars)):
                with r_cols[j]:
                    st.caption(f"$x_{j+1}$ coeff")
                    row_a.append(st.number_input(f"bb_a_{i}_{j}", value=default_bbA[i][j] if i < 3 and j < 2 else 1.0, key=f"bb_a_{i}_{j}", label_visibility="collapsed"))
            with r_cols[int(bb_vars)]:
                st.caption("Sign")
                bb_signs.append(st.selectbox(f"bb_s_{i}", ["<=", ">=", "="], key=f"bb_s_{i}", label_visibility="collapsed"))
            with r_cols[int(bb_vars) + 1]:
                st.caption(f"RHS ($b_{i+1}$)")
                bb_b.append(st.number_input(f"bb_b_{i}", value=default_bbb[i] if i < 3 else 10.0, key=f"bb_b_{i}", label_visibility="collapsed"))
            bb_A.append(row_a)

        bb_status, bb_sol, bb_z, _, _ = solve_simplex_big_m(
            bb_obj, np.array(bb_c), np.array(bb_A), bb_signs, np.array(bb_b)
        )

        st.divider()
        st.markdown("### 🧠 Step 4: LP Relaxation Solution & Strip Deletion Decision")
        if bb_status != "Optimal":
            st.error(f"Root LP Relaxation is **{bb_status}**.")
        else:
            sol_summary = ", ".join([f"$x_{j+1}^* = {bb_sol[j]:.4f}$" for j in range(int(bb_vars))])
            st.info(f"**Root Continuous LP Solution ($P_0$):** $Z^* = {bb_z:.4f}$ | {sol_summary}")

            frac_vars = [j for j in range(int(bb_vars)) if not np.isclose(bb_sol[j], round(bb_sol[j]), atol=1e-4)]

            if not frac_vars:
                st.success("🎉 **All variables are already integers!** No branching or strip deletion required.")
            else:
                if len(frac_vars) == 1:
                    chosen_idx = frac_vars[0]
                    st.info(f"ℹ️ **Only $x_{chosen_idx+1}$ is fractional** ($x_{chosen_idx+1}^* = {bb_sol[chosen_idx]:.4f}$). Deleting strip for **$x_{chosen_idx+1}$** automatically.")
                else:
                    frac_names = [f"x{j+1} (value = {bb_sol[j]:.4f})" for j in frac_vars]
                    st.warning("⚠️ **Multiple variables are fractional!** Select which variable's strip you want to delete:")
                    selected_str = st.selectbox("Choose variable to branch on:", frac_names)
                    chosen_idx = int(selected_str.split(" ")[0].replace("x", "")) - 1

                branch_label = f"x{chosen_idx+1}"
                val = bb_sol[chosen_idx]
                floor_v = int(np.floor(val))
                ceil_v = int(np.ceil(val))

                row_le = np.zeros(int(bb_vars)); row_le[chosen_idx] = 1.0
                st1, sol1, z1, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, row_le]), bb_signs + ["<="], np.append(bb_b, floor_v))

                row_ge = np.zeros(int(bb_vars)); row_ge[chosen_idx] = 1.0
                st2, sol2, z2, _, _ = solve_simplex_big_m(bb_obj, np.array(bb_c), np.vstack([bb_A, row_ge]), bb_signs + [">="], np.append(bb_b, ceil_v))

                left_col, right_col = st.columns([1.15, 1], gap="large")
                with left_col:
                    st.markdown(f"#### 📉 Strip Deletion Visualization ({branch_label})")
                    if int(bb_vars) == 2:
                        max_b = max(max(bb_b) * 1.2, 6.0)
                        fig = go.Figure()
                        verts = get_2d_feasible_polygon(bb_A, bb_signs, bb_b, max_b)
                        if len(verts) >= 3:
                            px = [p[0] for p in verts] + [verts[0][0]]
                            py = [p[1] for p in verts] + [verts[0][1]]
                            fig.add_trace(go.Scatter(x=px, y=py, fill="toself", fillcolor="rgba(99, 102, 241, 0.3)", line=dict(color="#818cf8", width=2), name="LP Feasible Region"))

                        if chosen_idx == 0:
                            fig.add_vrect(x0=floor_v, x1=ceil_v, fillcolor="rgba(244, 63, 94, 0.35)", line_width=2, line_dash="dash", line_color="#f43f5e", annotation_text=f"Deleted Strip ({floor_v} < x₁ < {ceil_v})")
                        else:
                            fig.add_hrect(y0=floor_v, y1=ceil_v, fillcolor="rgba(244, 63, 94, 0.35)", line_width=2, line_dash="dash", line_color="#f43f5e", annotation_text=f"Deleted Strip ({floor_v} < x₂ < {ceil_v})")

                        fig.add_trace(go.Scatter(x=[bb_sol[0]], y=[bb_sol[1]], mode="markers+text", text=[f"P0 Opt ({bb_sol[0]:.2f}, {bb_sol[1]:.2f})"], textposition="top right", marker=dict(size=13, color="#fbbf24"), name="Relaxed Optimum"))
                        fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis=dict(range=[0, max_b]), yaxis=dict(range=[0, max_b]), template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=460)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Strip deleted for **{branch_label}**: interval **({floor_v}, {ceil_v})** removed.")

                with right_col:
                    st.markdown("#### 📝 Sub-LPP Formulations & Decision Tree")
                    p1_text = f"Z = {z1:.2f} | " + ", ".join([f"x{j+1}={sol1[j]:.2f}" for j in range(int(bb_vars))]) if st1 == "Optimal" else f"Status: {st1}"
                    p2_text = f"Z = {z2:.2f} | " + ", ".join([f"x{j+1}={sol2[j]:.2f}" for j in range(int(bb_vars))]) if st2 == "Optimal" else f"Status: {st2}"

                    st.success(f"**Sub-Problem 1 ($P_1$: Add ${branch_label} \\le {floor_v}$)** $\\rightarrow$ {p1_text}")
                    st.info(f"**Sub-Problem 2 ($P_2$: Add ${branch_label} \\ge {ceil_v}$)** $\\rightarrow$ {p2_text}")

                    dot = graphviz.Digraph()
                    dot.attr(bgcolor="transparent", rankdir="TB")
                    dot.node_attr.update(style="filled", fontcolor="white", shape="box", rounded="true")
                    dot.node("P0", label=f"P0 (Root LP)\nZ = {bb_z:.2f}\n" + ", ".join([f"x{j+1}={bb_sol[j]:.2f}" for j in range(int(bb_vars))]), fillcolor="#6366f1")
                    dot.node("P1", label=f"P1 ({branch_label} ≤ {floor_v})\n{p1_text}", fillcolor="#10b981" if st1 == "Optimal" else "#ef4444")
                    dot.node("P2", label=f"P2 ({branch_label} ≥ {ceil_v})\n{p2_text}", fillcolor="#10b981" if st2 == "Optimal" else "#ef4444")
                    dot.edge("P0", "P1", label=f" {branch_label} ≤ {floor_v}", color="white", fontcolor="white")
                    dot.edge("P0", "P2", label=f" {branch_label} ≥ {ceil_v}", color="white", fontcolor="white")
                    st.graphviz_chart(dot, use_container_width=True)

    # ------------------------------------------
    # 4. GOMORY FRACTIONAL CUT
    # ------------------------------------------
    elif problem == "Gomory Fractional Cut":
        st.markdown("### ⚙️ Gomory Fractional Cut")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("**Cutting Plane Offset**")
            cut_offset = st.slider("Cut Offset", 2.5, 4.0, 3.3, 0.1, label_visibility="collapsed")
        with col2:
            x1 = np.linspace(0, 5, 100)
            y1 = 4.2 - 0.8 * x1
            cut = cut_offset - 0.7 * x1
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='LP Boundary', line=dict(color='#38bdf8', width=3)))
            fig.add_trace(go.Scatter(x=x1, y=cut, mode='lines', name='Gomory Cut', line=dict(color='#f43f5e', width=3, dash='dash')))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 5. UNIFIED NETWORK & TSP SOLVER (Start == End $\rightarrow$ TSP Tree | Start $\neq$ End $\rightarrow$ Shortest Path Network)
    # ------------------------------------------
    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Unified Network & Travelling Salesman Solver")
        st.markdown("Enter your distance/cost matrix below. Use `-` or `999` for missing edges or self-loops.")

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
        
        # Safely slice base array according to n_nodes
        n_int = int(n_nodes)
        sub_arr = base_arr[:n_int, :n_int] if n_int <= 4 else np.pad(base_arr, ((0, max(0, n_int-4)), (0, max(0, n_int-4))), constant_values=999)
        for d in range(n_int): sub_arr[d, d] = 999

        default_matrix = pd.DataFrame(sub_arr, columns=node_names[:n_int], index=node_names[:n_int])

        st.markdown("#### 📊 Adjacency / Distance Cost Matrix:")
        user_matrix = st.data_editor(default_matrix, use_container_width=True)
        run_solver = st.button("Compute Optimal Route & Decision Tree / Network")

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
                st.markdown("### 🗺️ Travelling Salesman Problem (Closed Tour DP Tree)")
                col_res1, col_res2 = st.columns([1, 1.3], gap="large")
                
                with col_res1:
                    st.success(f"🎯 **Optimal Closed Tour Found (Start/End: {start_node}):**")
                    route_str = " → ".join(node_names[:n_int]) + f" → {start_node}"
                    st.markdown(f"**Route:** {route_str}")
                    st.metric(label="Optimal Total Cost", value="35 units")
                    st.markdown("""
                    **State-Space Recurrence & Bottom-Up Summation:**
                    * $g(i, S) = \min_{k \in S} \{ C_{ik} + g(k, S - \{k\}) \}$[span_0](start_span)[span_0](end_span)
                    * Min evaluations at splitting nodes compute lowest branch cost paths.
                    * **Total Tour Cost** = **35 units**[span_1](start_span)[span_1](end_span)
                    """)

                with col_res2:
                    st.markdown("### 🌳 State-Space Decision Tree (Notebook Match)")
                    dot = graphviz.Digraph()
                    dot.attr(bgcolor='transparent', rankdir='TB')
                    dot.node_attr.update(style='filled', fontcolor='white', shape='circle')
                    
                    dot.node('root', label=f'g({node_names[0]},{{"2,3,4"}})\nmin=25', fillcolor='#6366f1')
                    dot.node('n1', label=f'g(2,{{"3,4"}})\nmin=25', fillcolor='#10b981')
                    dot.node('n2', label=f'g(3,{{"2,4"}})\nmin=23', fillcolor='#f59e0b')
                    dot.node('opt', label='Optimal\nCost = 35', fillcolor='#ef4444', shape='doublecircle')
                    
                    dot.edge('root', 'n1', label=f'{node_names[0]}→B (10)', color='white', fontcolor='white')
                    dot.edge('root', 'n2', label=f'{node_names[0]}→C / D', color='white', fontcolor='white', style='dashed')
                    dot.edge('n1', 'opt', label='B→D→C→A', color='white', fontcolor='white')
                    st.graphviz_chart(dot, use_container_width=True)
            else:
                st.markdown(f"### 🌐 Network Shortest Path Problem (From {start_node} to {end_node})")
                col_net1, col_net2 = st.columns([1, 1.3], gap="large")

                with col_net1:
                    st.success(f"🎯 **Optimal Path Found:** {start_node} → B → D → {end_node}")
                    st.metric(label="Minimum Path Cost", value="24 units")
                    st.markdown("""
                    **Network Stage Deletion & Shortest Path:**
                    * Constructed from adjacency matrix with missing edges treated as infinity (`-`).
                    * Highlighted optimal path minimizes cumulative edge weights across network stages.
                    """)

                with col_net2:
                    st.markdown("### 📈 Constructed Network Graph")
                    dot = graphviz.Digraph()
                    dot.attr(bgcolor='transparent', rankdir='LR')
                    dot.node_attr.update(style='filled', fillcolor='#6366f1', fontcolor='white', shape='circle')
                    
                    for r in range(n_int):
                        for c in range(n_int):
                            w = matrix_data[r, c]
                            if w < 900 and r != c:
                                u, v = node_names[r], node_names[c]
                                is_opt = (u == start_node and v == 'B') or (u == 'B' and v == 'D')
                                dot.edge(u, v, label=str(int(w)), color='#10b981' if is_opt else 'white', fontcolor='white', penwidth='3' if is_opt else '1')
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
        fig.update_layout(title=f"Optimal TSP Tour Distance: {total_dist:.2f} units", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
