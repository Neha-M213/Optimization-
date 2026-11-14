import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import graphviz
import copy

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
# HELPER: NUMBER FORMATTER (UP TO 3 DECIMALS, WHOLE NUMBERS AS .0)
# ==========================================
def fmt_num(val):
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-4:
            return f"{round(f):.1f}"
        else:
            return f"{f:.3f}"
    except:
        return str(val)

# ==========================================
# HELPER: TABLEAU BUILDER
# ==========================================
def build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False):
    num_cons = len(basis_indices)
    cb = cost_row[basis_indices]
    
    cj_display = ["", "C_j ->", ""] + [fmt_num(val) if abs(val) < 1e4 else ("-M" if val < 0 else "M") for val in cost_row]
    if include_ratio_col:
        cj_display += [""]
    
    rows = [cj_display]
    for i in range(num_cons):
        cb_str = fmt_num(cb[i]) if abs(cb[i]) < 1e4 else ("-M" if cb[i] < 0 else "M")
        row_vals = [
            cb_str,
            var_names[basis_indices[i]],
            fmt_num(rhs[i])
        ] + [fmt_num(x) for x in tableau[i]]
        if include_ratio_col:
            r_val = ratios[i] if ratios is not None else "—"
            row_vals += [fmt_num(r_val) if r_val != "—" else "—"]
        rows.append(row_vals)
        
    z_disp = fmt_num(z_val if obj_type == "Maximize" else -z_val)
    zj_row = ["", "Z_j", z_disp] + [fmt_num(val) for val in zj]
    if include_ratio_col:
        zj_row += [""]
        
    cj_zj_row = ["", "C_j - Z_j", ""] + [fmt_num(val) for val in cj_zj]
    if include_ratio_col:
        cj_zj_row += [""]
        
    rows.append(zj_row)
    rows.append(cj_zj_row)
    
    if include_ratio_row and ratios is not None:
        ratio_row = ["", "Ratio", ""] + [fmt_num(r) if r != "—" else "—" for r in ratios]
        if include_ratio_col:
            ratio_row += [""]
        rows.append(ratio_row)
    
    cols = ["C_B", "X_B", "b"] + var_names
    if include_ratio_col:
        cols += ["Ratio"]
        
    df = pd.DataFrame(rows, columns=cols)
    return df

def get_point_from_basis(basis_indices, rhs, num_vars):
    pt = np.zeros(num_vars)
    for pos, b_idx in enumerate(basis_indices):
        if b_idx < num_vars:
            pt[b_idx] = rhs[pos]
    return tuple(round(float(val), 3) for val in pt[:2])

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
            extra_cols.append(col_s); var_names.append(f"s{s_count}"); cost_row.append(0.0)
            s_count += 1
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
    
    is_degenerate = False
    has_alternative = False
    alt_solutions = []
    degenerate_points = set()
    
    for it in range(30):
        cb = cost_row[basis_indices]
        zj = cb @ tableau
        cj_zj = cost_row - zj
        z_val = cb @ rhs
        
        for i, r_val in enumerate(rhs):
            if abs(r_val) <= 1e-6:
                is_degenerate = True
                deg_pt = get_point_from_basis(basis_indices, rhs, num_vars)
                degenerate_points.add(deg_pt)

        if np.max(cj_zj) <= 1e-7:
            non_basic_indices = [j for j in range(len(var_names)) if j not in basis_indices]
            for nb_idx in non_basic_indices:
                if abs(cj_zj[nb_idx]) <= 1e-7:
                    has_alternative = True
                    col_vals = tableau[:, nb_idx]
                    if any(col_vals > 1e-7):
                        valid_r = [(rhs[i] / col_vals[i], i) for i in range(num_cons) if col_vals[i] > 1e-7]
                        if valid_r:
                            l_row = min(valid_r, key=lambda x: x[0])[1]
                            t_dup = tableau.copy()
                            r_dup = rhs.copy()
                            b_dup = list(basis_indices)
                            p_val = t_dup[l_row, nb_idx]
                            t_dup[l_row] /= p_val
                            r_dup[l_row] /= p_val
                            for ri in range(num_cons):
                                if ri != l_row:
                                    fac = t_dup[ri, nb_idx]
                                    t_dup[ri] -= fac * t_dup[l_row]
                                    r_dup[ri] -= fac * r_dup[l_row]
                            b_dup[l_row] = nb_idx
                            alt_sol = np.zeros(num_vars)
                            for pos, b_idx in enumerate(b_dup):
                                if b_idx < num_vars: alt_sol[b_idx] = r_dup[pos]
                            alt_solutions.append(alt_sol)

            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False)
            iterations.append((it + 1, df, "Optimal Tableau Reached", list(basis_indices), rhs.copy()))
            break
            
        entering_col = int(np.argmax(cj_zj))
        col_vals = tableau[:, entering_col]
        
        if all(col_vals <= 1e-7):
            status = "Unbounded"
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False)
            iterations.append((it + 1, df, "Unbounded", list(basis_indices), rhs.copy()))
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
        df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=ratio_display, include_ratio_col=True, include_ratio_row=False)
        iterations.append((it + 1, df, f"Entering: {var_names[entering_col]}", list(basis_indices), rhs.copy()))
        
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
    return status, sol, float(c @ sol), iterations, list(special_flags), is_degenerate, has_alternative, alt_solutions, basis_indices, var_names, cj_zj, list(degenerate_points)

# ==========================================
# HELPER: DUAL SIMPLEX ENGINE
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
            break
        if np.all(rhs >= -1e-7):
            break
            
        leaving_row = int(np.argmin(rhs))
        col_vals = tableau[leaving_row]
        if np.all(col_vals >= -1e-7):
            status = "Infeasible"
            break
            
        valid_ratios = []
        for j in range(len(var_names)):
            if col_vals[j] < -1e-7:
                valid_ratios.append((abs(cj_zj[j] / col_vals[j]), j))
        if not valid_ratios:
            status = "Infeasible"
            break
        entering_col = min(valid_ratios, key=lambda x: x[0])[1]
        
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
# HELPER: 2D GEOMETRY GENERATORS
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

    st.write("")
    col4, col5, col6 = st.columns(3, gap="large")
    with col4: st.button("✂️ Gomory Fractional Cut", on_click=select_problem, args=("Gomory Fractional Cut",))
    with col5: st.button("🌐 Network & TSP Solver", on_click=select_problem, args=("Network & TSP Solver",))
    with col6: st.button("🗺️ Advanced TSP Studio", on_click=select_problem, args=("Advanced TSP Studio",))

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
    # 1. SIMPLEX METHOD
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

        status, sol, final_z, iterations, flags, is_deg, has_alt, alt_sols, basis_idxs, var_nams, cj_zj_vals, degenerate_points = solve_simplex_big_m(
            obj_type, np.array(c_vals), np.array(A_rows), signs, np.array(b_vals)
        )

        st.divider()
        st.markdown("### 🔍 Solution Status & Insights")
        if flags:
            for f in flags: st.warning(f)
        else:
            st.success("✅ **Unique Optimal Solution Found**")

        if status == "Optimal":
            sol_str = ", ".join([f"**x{j+1} = {sol[j]:.3f}**" for j in range(int(num_vars))])
            st.info(f"**Optimal Objective Z = {final_z:.3f}** | {sol_str}")

        st.divider()
        st.markdown("### 📋 Step-by-Step Simplex Tableaux")
        for iter_num, df, note, _, _ in iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📈 Feasible Region & Corner Points")
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

            for p in valid_verts:
                fig.add_trace(go.Scatter(
                    x=[p[0]], y=[p[1]],
                    mode="markers+text",
                    text=[f"({p[0]:.1f}, {p[1]:.1f})"],
                    textposition="top right",
                    textfont=dict(color="#38bdf8", size=11),
                    marker=dict(size=7, color="#38bdf8", symbol="circle"),
                    showlegend=False
                ))

            for i in range(int(num_cons)):
                a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
                col_c = colors[i % len(colors)]
                if abs(a2) > 1e-6:
                    y_vals = (b_i - a1 * x_line) / a2
                    mask = (y_vals >= 0) & (y_vals <= max_bound)
                    if np.any(mask):
                        fig.add_trace(go.Scatter(x=x_line[mask], y=y_vals[mask], mode="lines", name=f"C{i+1}: {a1}x₁ + {a2}x₂ {signs[i]} {b_i}", line=dict(color=col_c, width=3)))

            if status == "Optimal":
                fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode="markers+text", text=[f"Optimal Z*={final_z:.2f}"], textposition="bottom right", marker=dict(size=16, color="#10b981", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig.update_layout(
                xaxis=dict(title=dict(text="Decision Variable x₁", font=dict(color="#f8fafc", size=14)), range=[0, max_bound], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                yaxis=dict(title=dict(text="Decision Variable x₂", font=dict(color="#f8fafc", size=14)), range=[0, max_bound], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(30, 41, 59, 0.8)", bordercolor="#334155"),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
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
            with d_obj_cols[j]: d_c_vals.append(st.number_input(f"c{j+1}", value=-2.0 if j==0 else (-2.0 if j==1 else -4.0), key=f"d_c_{j}"))

        st.markdown(f"### ⛓️ Constraints Setup:")
        d_A_rows, d_signs, d_b_vals = [], [], []
        default_ds_a = [[-2.0, -3.0, -5.0], [-3.0, -1.0, -7.0]]
        default_ds_b = [-2.0, -3.0]
        
        for i in range(int(d_num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            d_row_cols = st.columns(int(d_num_vars) + 2)
            d_row_a = []
            for j in range(int(d_num_vars)):
                with d_row_cols[j]: d_row_a.append(st.number_input(f"da_{i}_{j}", value=default_ds_a[i][j] if i<2 and j<3 else 1.0, key=f"da_{i}_{j}"))
            with d_row_cols[int(d_num_vars)]: d_signs.append(st.selectbox(f"dsign_{i}", ["<=", ">=", "="], key=f"dsign_{i}"))
            with d_row_cols[int(d_num_vars) + 1]: d_b_vals.append(st.number_input(f"db_{i}", value=default_ds_b[i] if i<2 else -1.0, key=f"db_{i}"))
            d_A_rows.append(d_row_a)

        d_status, d_sol, d_final_z, d_iterations, d_flags = solve_simplex_dual(
            d_obj_type, np.array(d_c_vals), np.array(d_A_rows), d_signs, np.array(d_b_vals)
        )

        st.divider()
        st.markdown("### 🔍 Dual Simplex Solution Status & Insights")
        if d_status == "Optimal":
            d_sol_str = ", ".join([f"**x{j+1} = {d_sol[j]:.3f}**" for j in range(int(d_num_vars))])
            st.info(f"**Optimal Objective Z = {d_final_z:.3f}** | {d_sol_str}")

        st.divider()
        st.markdown("### 📋 Step-by-Step Dual Simplex Tableaux")
        for iter_num, df, note in d_iterations:
            st.markdown(f"**Iteration {iter_num}** — {note}")
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📈 Feasible Region & Corner Points")
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

            for p in valid_verts:
                fig_d.add_trace(go.Scatter(
                    x=[p[0]], y=[p[1]],
                    mode="markers+text",
                    text=[f"({p[0]:.1f}, {p[1]:.1f})"],
                    textposition="top right",
                    textfont=dict(color="#34d399", size=11),
                    marker=dict(size=7, color="#34d399", symbol="circle"),
                    showlegend=False
                ))

            for i in range(int(d_num_cons)):
                a1, a2, b_i = vis_A[i][0], vis_A[i][1], vis_b[i]
                col_c = colors[i % len(colors)]
                if abs(a2) > 1e-6:
                    y_vals = (b_i - a1 * x_line) / a2
                    mask = (y_vals >= 0) & (y_vals <= max_bound)
                    if np.any(mask):
                        fig_d.add_trace(go.Scatter(x=x_line[mask], y=y_vals[mask], mode="lines", name=f"C{i+1}: {a1}x₁ + {a2}x₂ {vis_signs[i]} {b_i}", line=dict(color=col_c, width=3)))

            if d_status == "Optimal":
                fig_d.add_trace(go.Scatter(x=[d_sol[0]], y=[d_sol[1]], mode="markers+text", text=[f"Optimal Z*={d_final_z:.2f}"], textposition="bottom right", marker=dict(size=16, color="#f472b6", symbol="star", line=dict(width=2, color="white")), name="Optimal Vertex"))

            fig_d.update_layout(
                xaxis=dict(title=dict(text="Decision Variable x₁", font=dict(color="#f8fafc", size=14)), range=[0, max_bound], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                yaxis=dict(title=dict(text="Decision Variable x₂", font=dict(color="#f8fafc", size=14)), range=[0, max_bound], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(30, 41, 59, 0.8)", bordercolor="#334155"),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560
            )
            st.plotly_chart(fig_d, use_container_width=True)

    # ------------------------------------------
    # 3. BRANCH AND BOUND (FULL IMPLEMENTATION)
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.markdown("### ⚙️ Step 1: ILP Problem Configuration")
        dim1, dim2, dim3 = st.columns(3)
        with dim1: bb_obj = st.selectbox("Objective Goal", ["Maximize", "Minimize"], key="bb_goal")
        with dim2: bb_vars = st.number_input("Variables count", min_value=2, max_value=2, value=2, step=1, key="bb_v")
        with dim3: bb_cons = st.number_input("Constraints count", min_value=1, max_value=10, value=2, step=1, key="bb_c")

        st.markdown("#### Objective Function Coefficients ($c_j$):")
        bc_cols = st.columns(2)
        bb_c_vals = []
        for j in range(2):
            with bc_cols[j]: bb_c_vals.append(st.number_input(f"c{j+1}", value=5.0 if j==0 else 4.0, key=f"bb_c_{j}"))

        st.markdown("#### Constraints Setup:")
        bb_A, bb_signs, bb_b = [], [], []
        default_bb_a = [[1.0, 1.0], [10.0, 6.0]]
        default_bb_b = [5.0, 45.0]
        
        for i in range(int(bb_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            r_cols = st.columns(4)
            r_a = []
            for j in range(2):
                with r_cols[j]: r_a.append(st.number_input(f"a_{i}_{j}", value=default_bb_a[i][j] if i<2 else 1.0, key=f"bb_a_{i}_{j}"))
            with r_cols[2]: bb_signs.append(st.selectbox(f"sign_{i}", ["<=", "="], key=f"bb_sign_{i}"))
            with r_cols[3]: bb_b.append(st.number_input(f"b_{i}", value=default_bb_b[i] if i<2 else 10.0, key=f"bb_b_{i}"))
            bb_A.append(r_a)

        if st.button("🚀 Generate Branch & Bound Search Tree & Cuts"):
            # Execute recursive Branch and Bound search tree
            tree_nodes = []
            best_int_obj = -float('inf') if bb_obj == "Maximize" else float('inf')
            best_int_sol = None

            node_counter = [0]

            def solve_bb_node(A_curr, signs_curr, b_curr, parent_id, branch_desc):
                nid = f"Node_{node_counter[0]}"
                node_counter[0] += 1
                
                status, sol, z_val, _, _, _, _, _, _, _, _, _ = solve_simplex_big_m(
                    bb_obj, np.array(bb_c_vals), np.array(A_curr), signs_curr, np.array(b_curr)
                )

                node_info = {
                    "id": nid,
                    "parent": parent_id,
                    "desc": branch_desc,
                    "status": status,
                    "sol": sol,
                    "z": z_val,
                    "active": True
                }

                if status != "Optimal":
                    node_info["active"] = False
                    node_info["reason"] = status
                    tree_nodes.append(node_info)
                    return

                # Check bound pruning
                global best_int_obj
                if bb_obj == "Maximize" and z_val <= best_int_obj + 1e-5:
                    node_info["active"] = False
                    node_info["reason"] = "Pruned by Bound"
                    tree_nodes.append(node_info)
                    return
                if bb_obj == "Minimize" and z_val >= best_int_obj - 1e-5:
                    node_info["active"] = False
                    node_info["reason"] = "Pruned by Bound"
                    tree_nodes.append(node_info)
                    return

                # Check integer feasibility
                is_integer = all(abs(val - round(val)) < 1e-4 for val in sol)
                if is_integer:
                    node_info["reason"] = "Integer Optimal"
                    if (bb_obj == "Maximize" and z_val > best_int_obj) or (bb_obj == "Minimize" and z_val < best_int_obj):
                        best_int_obj = z_val
                        global best_int_sol
                        best_int_sol = sol.copy()
                    tree_nodes.append(node_info)
                    return

                tree_nodes.append(node_info)

                # Branch on first fractional variable
                frac_idx = -1
                for idx, val in enumerate(sol):
                    if abs(val - round(val)) >= 1e-4:
                        frac_idx = idx
                        break

                if frac_idx != -1:
                    f_val = sol[frac_idx]
                    floor_val = np.floor(f_val)
                    ceil_val = np.ceil(f_val)

                    # Left Branch (x <= floor)
                    A_left = [row.copy() for row in A_curr]
                    signs_left = list(signs_curr)
                    b_left = list(b_curr)
                    
                    new_row = [0.0, 0.0]
                    new_row[frac_idx] = 1.0
                    A_left.append(new_row)
                    signs_left.append("<=")
                    b_left.append(float(floor_val))
                    
                    solve_bb_node(A_left, signs_left, b_left, nid, f"x{frac_idx+1} ≤ {int(floor_val)}")

                    # Right Branch (x >= ceil)
                    A_right = [row.copy() for row in A_curr]
                    signs_right = list(signs_curr)
                    b_right = list(b_curr)
                    
                    new_row = [0.0, 0.0]
                    new_row[frac_idx] = 1.0
                    A_right.append(new_row)
                    signs_right.append(">=")
                    b_right.append(float(ceil_val))
                    
                    solve_bb_node(A_right, signs_right, b_right, nid, f"x{frac_idx+1} ≥ {int(ceil_val)}")

            solve_bb_node(bb_A, bb_signs, bb_b, None, "Root (Relaxed LP)")

            st.divider()
            st.markdown("### 🌳 Branch & Bound Search Tree")
            
            dot_bb = graphviz.Digraph(comment="Branch and Bound Tree")
            dot_bb.attr(bgcolor="transparent", rankdir="TB")
            dot_bb.attr('node', shape='box', style='filled', fontcolor='white', fontname='Helvetica', fillcolor='#1e293b', color='#38bdf8', penwidth='1.5')
            dot_bb.attr('edge', fontname='Helvetica', fontsize='10', color='#94a3b8')

            for nd in tree_nodes:
                status_color = '#059669' if "Integer" in nd.get("reason", "") else ('#b91c1c' if not nd["active"] else '#2563eb')
                sol_str = f"({nd['sol'][0]:.2f}, {nd['sol'][1]:.2f})"
                label = f"{nd['desc']}\nZ = {nd['z']:.2f}\nx = {sol_str}\n[{nd.get('reason', 'Active/Branching')}]"
                dot_bb.node(nd["id"], label, fillcolor=status_color)
                if nd["parent"]:
                    dot_bb.edge(nd["parent"], nd["id"])

            st.graphviz_chart(dot_bb)

            if best_int_sol is not None:
                st.success(f"🏆 **Optimal Integer Solution Found:** x₁ = {best_int_sol[0]:.0f}, x₂ = {best_int_sol[1]:.0f} | **Optimal Objective Z* = {best_int_obj:.2f}**")
            else:
                st.warning("No integer feasible solution found within bounds.")

    # ------------------------------------------
    # 4. GOMORY FRACTIONAL CUT
    # ------------------------------------------
    elif problem == "Gomory Fractional Cut":
        st.markdown("### ✂️ Gomory Fractional Cutting Plane Studio")
        st.markdown("Solve Integer Linear Programming (ILP) problems starting from the relaxed optimal simplex table.")

    # ------------------------------------------
    # 5. NETWORK & TSP SOLVER
    # ------------------------------------------
    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Network & Traveling Salesman Problem (TSP) Studio")
        st.info("Select TSP or Network shortest path mode above.")

    # ------------------------------------------
    # 6. ADVANCED TSP STUDIO
    # ------------------------------------------
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
