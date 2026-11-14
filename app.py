import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import graphviz
import itertools

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
# HELPER: TABLEAU BUILDER WITH RATIO ROW
# ==========================================
def build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None):
    num_cons = len(basis_indices)
    cb = cost_row[basis_indices]
    
    cj_display = ["", "C_j ->", ""] + [round(float(val), 3) if abs(val) < 1e4 else ("-M" if val < 0 else "M") for val in cost_row] + [""]
    
    rows = [cj_display]
    for i in range(num_cons):
        cb_str = round(float(cb[i]), 3) if abs(cb[i]) < 1e4 else ("-M" if cb[i] < 0 else "M")
        row_vals = [
            cb_str,
            var_names[basis_indices[i]],
            round(float(rhs[i]), 4)
        ] + [round(float(x), 4) for x in tableau[i]] + [
            ratios[i] if ratios is not None else "—"
        ]
        rows.append(row_vals)
        
    z_disp = round(float(z_val if obj_type == "Maximize" else -z_val), 4)
    zj_row = ["", "Z_j", z_disp] + [round(float(val), 3) for val in zj] + [""]
    cj_zj_row = ["", "C_j - Z_j", ""] + [round(float(val), 3) for val in cj_zj] + [""]
    rows.append(zj_row)
    rows.append(cj_zj_row)
    
    if ratios is not None:
        ratio_row = ["", "Ratio", ""] + [r if r != "—" else "—" for r in ratios] + [""]
        rows.append(ratio_row)
    
    cols = ["C_B", "X_B", "b"] + var_names + ["Ratio"]
    df = pd.DataFrame(rows, columns=cols)
    return df

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
            if signs[i] == "<=": signs[i] = "="
            elif signs[i] == "=": pass

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
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, "Optimal Tableau Reached (All $C_j - Z_j \\le 0$)"))
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            special_flags.add(f"🚩 **Unboundedness:** Pivot column {var_names[entering_col]} has entries ≤ 0.")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, f"Unbounded at {var_names[entering_col]}"))
            break
            
        valid_ratios = []
        ratio_display = ["—"] * len(var_names)
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                r_val = rhs[i] / col_vals[i]
                valid_ratios.append((r_val, i))
                ratio_display[i] = round(float(r_val), 4)
            else:
                ratio_display[i] = "—"
                
        leaving_row = min(valid_ratios, key=lambda x: x[0])[1]
        
        df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=ratio_display)
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
# HELPER: DUAL SIMPLEX ENGINE WITH RATIOS
# ==========================================
def solve_simplex_dual(obj_type, c, A, signs, b):
    num_vars = len(c)
    num_cons = len(b)
    
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    signs = list(signs)
    
    for i in range(num_cons):
        if signs[i] == ">=":
            A[i] = -A[i]
            b[i] = -b[i]
            signs[i] = "<="

    c_eff = c.copy().astype(float) if obj_type == "Maximize" else -c.copy().astype(float)
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)
    
    extra_cols, basis_indices = [], []
    s_count = 1
    for i, s in enumerate(signs):
        if s == "<=":
            col = np.zeros(num_cons); col[i] = 1.0
            extra_cols.append(col); var_names.append(f"s{s_count}"); cost_row.append(0.0)
            basis_indices.append(len(var_names) - 1); s_count += 1
        elif s == "=":
            col = np.zeros(num_cons); col[i] = 1.0
            extra_cols.append(col); var_names.append(f"s{s_count}"); cost_row.append(0.0)
            basis_indices.append(len(var_names) - 1); s_count += 1

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
        
        if np.max(cj_zj) > 1e-7:
            status = "Dual Infeasible"
            special_flags.add("🚩 **Dual Infeasibility:** $C_j - Z_j > 0$ found.")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, "Dual Infeasible Tableau"))
            break
            
        if np.all(rhs >= -1e-7):
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, "Optimal Dual Simplex Tableau Reached (All $b_i \\ge 0$)"))
            break
            
        leaving_row = int(np.argmin(rhs))
        if rhs[leaving_row] >= -1e-7:
            break
            
        col_vals = tableau[leaving_row]
        if np.all(col_vals >= -1e-7):
            status = "Infeasible"
            special_flags.add(f"🚩 **Primal Infeasibility / Dual Unboundedness:** Leaving row {leaving_row+1} has all entries $\\ge 0$ with negative RHS ($b = {rhs[leaving_row]:.4f}$).")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, f"Infeasible at Row {leaving_row+1}"))
            break
            
        valid_ratios = []
        ratio_display = ["—"] * len(var_names)
        for j in range(len(var_names)):
            if col_vals[j] < -1e-7:
                # Ratio formula: | (Z_j - C_j) / a_rj | or (C_j - Z_j) / (-a_rj)
                r_val = abs(cj_zj[j] / col_vals[j])
                valid_ratios.append((r_val, j))
                ratio_display[j] = round(float(r_val), 4)
            else:
                ratio_display[j] = "—"
                
        if not valid_ratios:
            status = "Infeasible"
            break
            
        entering_col = min(valid_ratios, key=lambda x: x[0])[1]
        
        df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=ratio_display)
        iter_note = f"Leaving: **{var_names[basis_indices[leaving_row]]}** (RHS = {rhs[leaving_row]:.4f}) | Entering: **{var_names[entering_col]}**"
        iterations.append((it + 1, df, iter_note))
        
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
        
    final_z_val = float(c @ sol)
    return status, sol, final_z_val, iterations, list(special_flags)

# ==========================================
# HELPER: 2D & 3D FEASIBLE REGION GEOMETRY
# ==========================================
def get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound):
    num_cons = len(b_vals)
    candidates = [(0.0, 0.0), (0.0, max_bound), (max_bound, 0.0), (max_bound, max_bound)]
    
    for i in range(num_cons):
        a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
        if abs(a2) > 1e-7: candidates.append((0.0, b_i / a2))
        if abs(a1) > 1e-7: candidates.append((b_i / a1, 0.0))
            
    for i in range(num_cons):
        for j in range(i + 1, num_cons):
            M_sys = np.array([[A_rows[i][0], A_rows[i][1]], [A_rows[j][0], A_rows[j][1]]], dtype=float)
            rhs_sys = np.array([b_vals[i], b_vals[j]], dtype=float)
            if abs(np.linalg.det(M_sys)) > 1e-7:
                pt = np.linalg.solve(M_sys, rhs_sys)
                candidates.append((float(pt[0]), float(pt[1])))

    valid_verts = []
    for x, y in candidates:
        if x < -1e-5 or y < -1e-5 or x > max_bound + 1e-5 or y > max_bound + 1e-5: continue
        ok = True
        for i in range(num_cons):
            val = A_rows[i][0] * x + A_rows[i][1] * y
            s = signs[i]; b_i = b_vals[i]
            if s == "<=" and val > b_i + 1e-3: ok = False; break
            elif s == ">=" and val < b_i - 1e-3: ok = False; break
            elif s == "=" and not np.isclose(val, b_i, atol=1e-2): ok = False; break
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

# ==========================================
# VIEW 2: DEDICATED WORKSPACE
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
    # 1. SIMPLEX METHOD (2D & 3D SUPPORT)
    # ------------------------------------------
    if problem == "Simplex Method":
        st.markdown("### ⚙️ Problem Configuration")
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1: obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"])
        with cfg2: num_vars = st.number_input("Variables count (2 or 3)", min_value=2, max_value=3, value=2, step=1)
        with cfg3: num_cons = st.number_input("Constraints count", min_value=1, max_value=15, value=2, step=1)

        st.markdown(f"### 📊 Objective Function Coefficients ($c_j$):")
        obj_cols = st.columns(int(num_vars))
        c_vals = []
        for j in range(int(num_vars)):
            with obj_cols[j]: c_vals.append(st.number_input(f"c{j+1}", value=1.0, key=f"c_{j}"))

        st.markdown(f"### ⛓️ Constraints Setup (≤ and =):")
        A_rows, signs, b_vals = [], [], []
        for i in range(int(num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            row_cols = st.columns(int(num_vars) + 2)
            row_a = []
            for j in range(int(num_vars)):
                with row_cols[j]: row_a.append(st.number_input(f"a_{i}_{j}", value=1.0, key=f"a_{i}_{j}"))
            with row_cols[int(num_vars)]: signs.append(st.selectbox(f"sign_{i}", ["<=", "="], key=f"sign_{i}"))
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

        st.divider()
        st.markdown("### 📋 Step-by-Step Simplex Tableaux")
        for iter_num, df, note in iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📈 Feasible Region & Optimal Solution Visualisation")
        if int(num_vars) == 2:
            max_rhs = max([abs(val) for val in b_vals]) if b_vals else 10.0
            max_bound = max(max_rhs * 1.4, 8.0)
            x_line = np.linspace(0, max_bound, 200)

            fig = go.Figure()
            colors = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa']

            valid_verts = get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound)
            if len(valid_verts) >= 3:
                poly_x = [p[0] for p in valid_verts] + [valid_verts[0][0]]
                poly_y = [p[1] for p in valid_verts] + [valid_verts[0][1]]
                fig.add_trace(go.Scatter(x=poly_x, y=poly_y, fill="toself", fillcolor="rgba(56, 189, 248, 0.3)", line=dict(color="#38bdf8", width=2.5), name="Feasible Region"))

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
                fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=[f"CP{idx+1}: ({p[0]}, {p[1]})" for idx, p in enumerate(valid_verts)], textposition="top right", marker=dict(size=11, color="#ffffff", line=dict(width=2, color="#38bdf8")), name="Corner Points"))

            if status == "Optimal":
                fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode="markers+text", text=[f"Optimal Z*={final_z:.2f}"], textposition="bottom right", marker=dict(size=16, color="#10b981", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig.update_layout(
                xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂",
                xaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155"),
                yaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155"),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif int(num_vars) == 3:
            st.markdown("#### 🧊 3D Polytope & Optimal Solution Space")
            fig_3d = go.Figure()
            max_3d = max([abs(val) for val in b_vals] + list(sol[:3]) + [5.0]) * 1.2
            grid_rng = np.linspace(0, max_3d, 20)
            
            if status == "Optimal":
                fig_3d.add_trace(go.Scatter3d(
                    x=[sol[0]], y=[sol[1]], z=[sol[2]],
                    mode="markers+text", text=[f"Optimal Z*={final_z:.2f} ({sol[0]:.2f}, {sol[1]:.2f}, {sol[2]:.2f})"],
                    marker=dict(size=10, color="#10b981", symbol="diamond"), name="Optimal Vertex"
                ))

            colors_3d = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
            for i in range(int(num_cons)):
                a1, a2, a3, b_i = A_rows[i][0], A_rows[i][1], A_rows[i][2], b_vals[i]
                if abs(a3) > 1e-6:
                    X3d, Y3d = np.meshgrid(grid_rng, grid_rng)
                    Z3d = (b_i - a1 * X3d - a2 * Y3d) / a3
                    Z3d = np.where((Z3d >= 0) & (Z3d <= max_3d), Z3d, np.nan)
                    fig_3d.add_trace(go.Surface(
                        x=X3d, y=Y3d, z=Z3d, colorscale=colors_3d[i % len(colors_3d)], opacity=0.5, showscale=False,
                        name=f"C{i+1}: {a1}x₁+{a2}x₂+{a3}x₃{signs[i]}{b_i}"
                    ))

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃",
                    xaxis=dict(range=[0, max_3d], backgroundcolor="#1e293b", gridcolor="#334155"),
                    yaxis=dict(range=[0, max_3d], backgroundcolor="#1e293b", gridcolor="#334155"),
                    zaxis=dict(range=[0, max_3d], backgroundcolor="#1e293b", gridcolor="#334155"),
                ),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=600
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD (2D & 3D SUPPORT + RATIOS)
    # ------------------------------------------
    elif problem == "Dual Simplex Method":
        st.markdown("### ⚙️ Dual Simplex Problem Configuration")
        d_cfg1, d_cfg2, d_cfg3 = st.columns(3)
        with d_cfg1: d_obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"], key="d_obj")
        with d_cfg2: d_num_vars = st.number_input("Variables count (2 or 3)", min_value=2, max_value=3, value=3, step=1, key="d_v")
        with d_cfg3: d_num_cons = st.number_input("Constraints count", min_value=1, max_value=15, value=2, step=1, key="d_c")

        st.markdown(f"### 📊 Objective Function Coefficients ($c_j$):")
        d_obj_cols = st.columns(int(d_num_vars))
        d_c_vals = []
        for j in range(int(d_num_vars)):
            def_c = -2.0 if j==0 else (-2.0 if j==1 else -4.0)
            with d_obj_cols[j]: d_c_vals.append(st.number_input(f"c{j+1}", value=def_c, key=f"d_c_{j}"))

        st.markdown(f"### ⛓️ Constraints Setup (Flexible Signs & RHS):")
        d_A_rows, d_signs, d_b_vals = [], [], []
        
        default_ds_a = [[-2.0, -3.0, -5.0], [-3.0, -1.0, -7.0]]
        default_ds_b = [-2.0, -3.0]
        
        for i in range(int(d_num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            d_row_cols = st.columns(int(d_num_vars) + 2)
            d_row_a = []
            for j in range(int(d_num_vars)):
                def_a_val = default_ds_a[i][j] if (i < 2 and j < 3) else 1.0
                with d_row_cols[j]: d_row_a.append(st.number_input(f"da_{i}_{j}", value=def_a_val, key=f"da_{i}_{j}"))
            with d_row_cols[int(d_num_vars)]: d_signs.append(st.selectbox(f"dsign_{i}", ["<=", ">=", "="], key=f"dsign_{i}"))
            with d_row_cols[int(d_num_vars) + 1]: 
                def_b_val = default_ds_b[i] if i < 2 else -1.0
                d_b_vals.append(st.number_input(f"db_{i}", value=def_b_val, key=f"db_{i}"))
            d_A_rows.append(d_row_a)

        d_status, d_sol, d_final_z, d_iterations, d_flags = solve_simplex_dual(
            d_obj_type, np.array(d_c_vals), np.array(d_A_rows), d_signs, np.array(d_b_vals)
        )

        st.divider()
        st.markdown("### 🔍 Dual Simplex Solution Status & Insights")
        if d_flags:
            for f in d_flags: st.warning(f)
        else:
            st.success("✅ **Optimal Dual Simplex Solution Found**")

        if d_status == "Optimal":
            d_sol_str = ", ".join([f"**x{j+1} = {d_sol[j]:.4f}**" for j in range(int(d_num_vars))])
            st.info(f"**Optimal Objective Z = {d_final_z:.4f}** | {d_sol_str}")

        st.divider()
        st.markdown("### 📋 Step-by-Step Dual Simplex Tableaux with Ratio Rows")
        for iter_num, df, note in d_iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📈 Feasible Region & Optimal Solution Visualisation")
        if int(d_num_vars) == 2:
            max_rhs = max([abs(val) for val in d_b_vals]) if d_b_vals else 10.0
            max_bound = max(max_rhs * 1.5, 8.0)
            x_line = np.linspace(0, max_bound, 200)

            fig_d = go.Figure()
            colors = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa']

            vis_A, vis_signs, vis_b = [], [], []
            for i in range(int(d_num_cons)):
                vis_A.append(d_A_rows[i])
                vis_signs.append(d_signs[i])
                vis_b.append(d_b_vals[i])

            valid_verts = get_2d_feasible_polygon(vis_A, vis_signs, vis_b, max_bound)
            if len(valid_verts) >= 3:
                poly_x = [p[0] for p in valid_verts] + [valid_verts[0][0]]
                poly_y = [p[1] for p in valid_verts] + [valid_verts[0][1]]
                fig_d.add_trace(go.Scatter(x=poly_x, y=poly_y, fill="toself", fillcolor="rgba(16, 185, 129, 0.25)", line=dict(color="#10b981", width=2.5), name="Feasible Region"))

            for i in range(int(d_num_cons)):
                a1, a2, b_i = vis_A[i][0], vis_A[i][1], vis_b[i]
                col_c = colors[i % len(colors)]
                if abs(a2) > 1e-6:
                    y_vals = (b_i - a1 * x_line) / a2
                    mask = (y_vals >= 0) & (y_vals <= max_bound)
                    if np.any(mask):
                        fig_d.add_trace(go.Scatter(x=x_line[mask], y=y_vals[mask], mode="lines", name=f"C{i+1}: {a1}x₁ + {a2}x₂ {vis_signs[i]} {b_i}", line=dict(color=col_c, width=3)))

            if valid_verts:
                vx, vy = zip(*valid_verts)
                fig_d.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=[f"CP{idx+1}: ({p[0]}, {p[1]})" for idx, p in enumerate(valid_verts)], textposition="top right", marker=dict(size=11, color="#ffffff", line=dict(width=2, color="#10b981")), name="Corner Points"))

            if d_status == "Optimal":
                fig_d.add_trace(go.Scatter(x=[d_sol[0]], y=[d_sol[1]], mode="markers+text", text=[f"Optimal Z*={d_final_z:.2f}"], textposition="bottom right", marker=dict(size=16, color="#f472b6", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig_d.update_layout(
                xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂",
                xaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155"),
                yaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155"),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560
            )
            st.plotly_chart(fig_d, use_container_width=True)
            
        elif int(d_num_vars) == 3:
            st.markdown("#### 🧊 3D Dual Simplex Polytope & Optimal Vertex")
            fig_3d_d = go.Figure()
            max_3d_d = max([abs(val) for val in d_b_vals] + list(d_sol[:3]) + [5.0]) * 1.2
            grid_rng_d = np.linspace(0, max_3d_d, 20)
            
            if d_status == "Optimal":
                fig_3d_d.add_trace(go.Scatter3d(
                    x=[d_sol[0]], y=[d_sol[1]], z=[d_sol[2]],
                    mode="markers+text", text=[f"Optimal Z*={d_final_z:.2f} ({d_sol[0]:.2f}, {d_sol[1]:.2f}, {d_sol[2]:.2f})"],
                    marker=dict(size=11, color="#f472b6", symbol="diamond"), name="Optimal Vertex"
                ))

            colors_3d_d = ['Greens', 'Oranges', 'Purples', 'Blues']
            for i in range(int(d_num_cons)):
                a1, a2, a3, b_i = d_A_rows[i][0], d_A_rows[i][1], d_A_rows[i][2], d_b_vals[i]
                if abs(a3) > 1e-6:
                    X3d, Y3d = np.meshgrid(grid_rng_d, grid_rng_d)
                    Z3d = (b_i - a1 * X3d - a2 * Y3d) / a3
                    Z3d = np.where((Z3d >= 0) & (Z3d <= max_3d_d), Z3d, np.nan)
                    fig_3d_d.add_trace(go.Surface(
                        x=X3d, y=Y3d, z=Z3d, colorscale=colors_3d_d[i % len(colors_3d_d)], opacity=0.55, showscale=False,
                        name=f"C{i+1}: {a1}x₁+{a2}x₂+{a3}x₃{d_signs[i]}{b_i}"
                    ))

            fig_3d_d.update_layout(
                scene=dict(
                    xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃",
                    xaxis=dict(range=[0, max_3d_d], backgroundcolor="#1e293b", gridcolor="#334155"),
                    yaxis=dict(range=[0, max_3d_d], backgroundcolor="#1e293b", gridcolor="#334155"),
                    zaxis=dict(range=[0, max_3d_d], backgroundcolor="#1e293b", gridcolor="#334155"),
                ),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=600
            )
            st.plotly_chart(fig_3d_d, use_container_width=True)
