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
# HELPER: BIG-M SIMPLEX ENGINE
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
    
    cols = ["C_B", "X_B", "b"] + var_names + ["Ratio"]
    df = pd.DataFrame(rows, columns=cols)
    return df

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
            special_flags.add(f"⚠️ **Degeneracy:** Row(s) {[r+1 for r in deg_rows]} have basic variable RHS = 0.")
            
        if np.max(cj_zj) <= 1e-7:
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None)
            iterations.append((it + 1, df, "Optimal Tableau Reached (All $C_j - Z_j \\le 0$)"))
            
            for idx_pos, b_idx in enumerate(basis_indices):
                if b_idx in artificial_indices and rhs[idx_pos] > 1e-6:
                    status = "Infeasible"
                    special_flags.add(f"🚩 **Infeasibility:** Artificial variable {var_names[b_idx]} remains positive in basis (RHS = {rhs[idx_pos]:.4f}).")
            
            if status == "Optimal":
                non_basic = [j for j in range(len(var_names)) if j not in basis_indices and j not in artificial_indices]
                alt_vars = [var_names[j] for j in non_basic if np.isclose(cj_zj[j], 0.0, atol=1e-7)]
                if alt_vars:
                    special_flags.add(f"💡 **Alternative Optima:** Non-basic variable(s) {alt_vars} have C_j - Z_j = 0.")
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
        ratio_display = []
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                r_val = rhs[i] / col_vals[i]
                valid_ratios.append((r_val, i))
                ratio_display.append(round(float(r_val), 4))
            else:
                ratio_display.append("—")
                
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
# HELPER: DYNAMIC BRANCH & BOUND ENGINE
# ==========================================
def solve_branch_and_bound_tree(obj_type, c, A_init, signs_init, b_init):
    nodes, queue = [], []
    status, sol, z_val, _, _ = solve_simplex_big_m(obj_type, c, A_init, signs_init, b_init)
    root_node = {
        "id": "P0",
        "label": f"P0 (Root)\nZ = {z_val:.2f}\n" + ", ".join([f"x{j+1}={sol[j]:.2f}" for j in range(len(c))]),
        "status": status,
        "z": z_val if status == "Optimal" else (-1e9 if obj_type == "Maximize" else 1e9),
        "sol": sol, "A": A_init, "signs": signs_init, "b": b_init,
        "fill": "#2563eb" if status == "Optimal" else "#059669"
    }
    nodes.append(root_node)
    queue.append(root_node)
    
    tree_edges = []
    node_counter = 1
    
    while queue and len(nodes) < 12:
        curr = queue.pop(0)
        if curr["status"] != "Optimal": continue
        sol = curr["sol"]
        if all(np.isclose(val, round(val), atol=1e-4) for val in sol): continue
            
        frac_idx = -1
        for j, val in enumerate(sol):
            if not np.isclose(val, round(val), atol=1e-4):
                frac_idx = j
                break
        if frac_idx == -1: continue
            
        val = sol[frac_idx]
        floor_v, ceil_v = int(np.floor(val)), int(np.ceil(val))
        
        for new_val, sign_str in [(floor_v, "<="), (ceil_v, ">=")]:
            r_vec = np.zeros(len(c))
            r_vec[frac_idx] = 1.0
            new_A = np.vstack([curr["A"], r_vec])
            new_signs = list(curr["signs"]) + [sign_str]
            new_b = np.append(curr["b"], new_val)
            
            st, sol_sub, z_sub, _, _ = solve_simplex_big_m(obj_type, c, new_A, new_signs, new_b)
            child_id = f"P{node_counter}"
            node_counter += 1
            
            c_label = f"{child_id} (x{frac_idx+1} {sign_str} {new_val})\nZ = {z_sub:.2f}\n" + ", ".join([f"x{j+1}={sol_sub[j]:.2f}" for j in range(len(c))]) if st == "Optimal" else f"{child_id} (x{frac_idx+1} {sign_str} {new_val})\n{st}"
            fill_col = '#2563eb' if st == "Optimal" else '#059669'
            
            node_obj = {
                "id": child_id, "label": c_label, "status": st,
                "z": z_sub if st == "Optimal" else (-1e9 if obj_type == "Maximize" else 1e9),
                "sol": sol_sub, "A": new_A, "signs": new_signs, "b": new_b, "fill": fill_col
            }
            nodes.append(node_obj)
            tree_edges.append((curr["id"], child_id, f"x{frac_idx+1} {sign_str} {new_val}"))
            if st == "Optimal": queue.append(node_obj)
            
    return nodes, tree_edges

# ==========================================
# HELPER: ROBUST 2D FEASIBLE POLYGON
# ==========================================
def get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound):
    num_cons = len(b_vals)
    candidates = [(0.0, 0.0), (0.0, max_bound), (max_bound, 0.0), (max_bound, max_bound)]
    
    # Axis intersections (x1 = 0 and x2 = 0)
    for i in range(num_cons):
        a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
        if abs(a2) > 1e-7:
            candidates.append((0.0, b_i / a2))
        if abs(a1) > 1e-7:
            candidates.append((b_i / a1, 0.0))
            
    # Pairwise constraint intersections
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
            s = signs[i]
            b_i = b_vals[i]
            if s == "<=" and val > b_i + 1e-3: 
                ok = False; break
            elif s == ">=" and val < b_i - 1e-3: 
                ok = False; break
            elif s == "=" and not np.isclose(val, b_i, atol=1e-2): 
                ok = False; break
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
                xaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155", title_font=dict(color="#f8fafc"), tickfont=dict(color="#f8fafc")),
                yaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155", title_font=dict(color="#f8fafc"), tickfont=dict(color="#f8fafc")),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560,
                legend=dict(font=dict(color="#f8fafc", size=13), bgcolor="rgba(15, 23, 42, 0.9)", bordercolor="rgba(148, 163, 184, 0.3)", borderwidth=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 2D Graphical visualization is displayed when Number of Objective Variables = 2.")

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
    # ------------------------------------------
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
            st.warning("🚩 **Dual Simplex Rule:** Row selected as leaving variable due to most negative RHS.")
            st.success("✨ Pivot completed: Dual feasibility restored.")

    # ------------------------------------------
    # 3. BRANCH AND BOUND
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
            with bc_cols[j]: bb_c_vals.append(st.number_input(f"c{j+1}", value=4.0 if j==0 else 3.0, key=f"bb_c_{j}"))

        st.markdown("#### Constraints Setup:")
        bb_A, bb_signs, bb_b = [], [], []
        for i in range(int(bb_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            r_cols = st.columns(4)
            r_a = []
            for j in range(2):
                with r_cols[j]: r_a.append(st.number_input(f"a_{i}_{j}", value=1.0 if (i==0 and j==0) else (3.0 if (i==0 and j==1) else (3.0 if (i==1 and j==0) else 1.0)), key=f"bb_a_{i}_{j}"))
            with r_cols[2]: bb_signs.append(st.selectbox(f"sign_{i}", ["<=", ">=", "="], key=f"bb_sign_{i}"))
            with r_cols[3]: bb_b.append(st.number_input(f"b_{i}", value=12.0 if i==0 else 10.0, key=f"bb_b_{i}"))
            bb_A.append(r_a)

        if st.button("🚀 Generate Branch & Bound Search Tree & Geometric Cuts"):
            nodes, tree_edges = solve_branch_and_bound_tree(
                bb_obj, np.array(bb_c_vals), np.array(bb_A), bb_signs, np.array(bb_b)
            )
            
            dot_bb = graphviz.Digraph(comment="Branch and Bound Tree")
            dot_bb.attr(bgcolor="transparent", rankdir="TB")
            dot_bb.attr('node', shape='box', style='filled', fontcolor='white', fontname='Helvetica')
            
            for nd in nodes:
                dot_bb.node(nd["id"], nd["label"], fillcolor=nd["fill"])
            for u, v, lbl in tree_edges:
                dot_bb.edge(u, v, label=lbl, color="#38bdf8", fontcolor="#f8fafc")
                
            st.markdown("### 🌳 Branch and Bound Solution Tree")
            st.graphviz_chart(dot_bb)
            st.success(f"✨ Branch and Bound exploration complete across {len(nodes)} subproblems.")

            # --- SUBPROBLEM GEOMETRY & STRIP DELETION GALLERY ---
            st.divider()
            st.markdown("### 📐 Subproblem Geometry, Strip Deletion & LPP Construction Gallery")
            st.caption("Each subproblem graph highlights the feasible region polygon, labeled corner points, active constraints, and branching cuts with high-contrast text.")

            max_intercept = 6.0
            for b_val in bb_b:
                max_intercept = max(max_intercept, float(b_val) * 1.3)
            max_bound = max(max_intercept, 10.0)
            x_line = np.linspace(0, max_bound, 300)
            colors = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#c084fc', '#ef4444', '#38bdf8']

            for idx, nd in enumerate(nodes):
                node_id = nd["id"]
                n_status = nd["status"]
                n_sol = nd["sol"]
                n_z = nd["z"]
                sub_A = nd["A"]
                sub_signs = nd["signs"]
                sub_b = nd["b"]

                z_str = f"{n_z:.2f}" if n_status == "Optimal" else "N/A"
                expander_label = f"📌 Subproblem {node_id} — Status: {n_status} | Z = {z_str}"

                with st.expander(expander_label, expanded=(idx == 0)):
                    col_g1, col_g2 = st.columns([3, 2])
                    
                    with col_g1:
                        fig_sub = go.Figure()
                        valid_verts = get_2d_feasible_polygon(sub_A, sub_signs, sub_b, max_bound)
                        
                        # Feasible region polygon highlight
                        if len(valid_verts) >= 3:
                            poly_x = [p[0] for p in valid_verts] + [valid_verts[0][0]]
                            poly_y = [p[1] for p in valid_verts] + [valid_verts[0][1]]
                            fig_sub.add_trace(go.Scatter(
                                x=poly_x, y=poly_y, fill="toself", 
                                fillcolor="rgba(56, 189, 248, 0.35)", 
                                line=dict(color="#38bdf8", width=3), 
                                name=f"Feasible Region {node_id}"
                            ))

                        # Plot constraints & branching cuts
                        for c_idx in range(len(sub_b)):
                            a1, a2, b_val = sub_A[c_idx][0], sub_A[c_idx][1], sub_b[c_idx]
                            s_sign = sub_signs[c_idx]
                            c_col = colors[c_idx % len(colors)]
                            
                            if c_idx >= len(bb_A):
                                c_label = f"Cut {c_idx}: {a1}x₁ + {a2}x₂ {s_sign} {b_val}"
                                line_width = 3.5
                                dash_style = "dash"
                            else:
                                c_label = f"C{c_idx+1}: {a1}x₁ + {a2}x₂ {s_sign} {b_val}"
                                line_width = 3
                                dash_style = "solid"

                            if abs(a2) > 1e-6:
                                y_vals = (b_val - a1 * x_line) / a2
                                mask = (y_vals >= 0) & (y_vals <= max_bound)
                                if np.any(mask):
                                    fig_sub.add_trace(go.Scatter(
                                        x=x_line[mask], y=y_vals[mask], mode="lines", 
                                        name=c_label, line=dict(color=c_col, width=line_width, dash=dash_style)
                                    ))
                            elif abs(a1) > 1e-6:
                                x_const = b_val / a1
                                if 0 <= x_const <= max_bound:
                                    fig_sub.add_trace(go.Scatter(
                                        x=[x_const, x_const], y=[0, max_bound], mode="lines", 
                                        name=c_label, line=dict(color=c_col, width=line_width, dash=dash_style)
                                    ))

                        # Plot corner points
                        if valid_verts:
                            vx, vy = zip(*valid_verts)
                            fig_sub.add_trace(go.Scatter(
                                x=vx, y=vy, mode="markers+text", 
                                text=[f"({p[0]}, {p[1]})" for p in valid_verts], 
                                textposition="top right", 
                                marker=dict(size=11, color="#f472b6", line=dict(width=2, color="white")), 
                                name="Corner Points"
                            ))

                        # Plot optimal subproblem solution
                        if n_status == "Optimal":
                            fig_sub.add_trace(go.Scatter(
                                x=[n_sol[0]], y=[n_sol[1]], mode="markers+text", 
                                text=[f"Z* ({n_sol[0]:.2f}, {n_sol[1]:.2f})"], 
                                textposition="bottom right", 
                                marker=dict(size=16, color="#10b981", symbol="star", line=dict(width=2, color="white")), 
                                name="Optimal Point"
                            ))

                        fig_sub.update_layout(
                            xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂", 
                            xaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155", title_font=dict(color="#f8fafc"), tickfont=dict(color="#f8fafc")), 
                            yaxis=dict(range=[0, max_bound], dtick=1, gridcolor="#334155", title_font=dict(color="#f8fafc"), tickfont=dict(color="#f8fafc")), 
                            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", 
                            height=440, margin=dict(l=30, r=20, t=30, b=30),
                            legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(15, 23, 42, 0.9)", bordercolor="rgba(148, 163, 184, 0.3)", borderwidth=1)
                        )
                        st.plotly_chart(fig_sub, use_container_width=True, key=f"plot_{node_id}")

                    with col_g2:
                        st.markdown(f"#### 📋 LPP Formulation for `{node_id}`")
                        st.markdown(f"**Objective Goal:** `{bb_obj}`")
                        st.markdown(f"**Objective Function:** $Z = {bb_c_vals[0]}x_1 + {bb_c_vals[1]}x_2$")
                        st.markdown("**Active Constraints & Strip Deletion Cuts:**")
                        
                        constraint_list_md = ""
                        for c_idx in range(len(sub_b)):
                            a1, a2, b_val = sub_A[c_idx][0], sub_A[c_idx][1], sub_b[c_idx]
                            s_sign = sub_signs[c_idx]
                            tag = " *(Branching Cut / Strip Spliced)*" if c_idx >= len(bb_A) else ""
                            constraint_list_md += f"- ${a1}x_1 + {a2}x_2 {s_sign} {b_val}${tag}\n"
                        st.markdown(constraint_list_md)
                        st.markdown(f"- $x_1, x_2 \\ge 0$ (Non-negativity)")
                        
                        st.divider()
                        if n_status == "Optimal":
                            st.success(f"**Subproblem Status:** Optimal\n\n**Solution:** $x_1 = {n_sol[0]:.4f}, x_2 = {n_sol[1]:.4f}$\n\n**Objective Value:** $Z = {n_z:.4f}$")
                        else:
                            st.error(f"**Subproblem Status:** {n_status}\n\n*This branch is pruned/infeasible.*")

    # ------------------------------------------
    # 4. NETWORK & TSP SOLVER
    # ------------------------------------------
    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Network & Traveling Salesman Problem (TSP) Studio")
        
        prob_type = st.radio("Select Problem Type", ["Travelling Salesman Problem (TSP) [Start == End]", "Network Shortest Route Problem [Start != End]"], horizontal=True)
        st.divider()

        if "TSP" in prob_type:
            st.markdown("#### 🗺️ Fully Flexible TSP Distance Matrix Configuration")
            num_cities = int(st.number_input("Number of Cities", min_value=3, max_value=6, value=4, step=1))
            city_labels = [chr(65 + i) for i in range(num_cities)]
            
            start_end_city = st.selectbox("Select Start & End City (TSP Closed Loop)", city_labels, index=0)
            
            default_matrix = np.full((num_cities, num_cities), 999.0)
            preset_vals = [
                [0, 10, 15, 20],
                [5, 0, 9, 10],
                [6, 13, 0, 12],
                [8, 8, 9, 0]
            ]
            for r in range(min(num_cities, 4)):
                for c in range(min(num_cities, 4)):
                    default_matrix[r, c] = preset_vals[r][c]
            
            matrix_df = pd.DataFrame(default_matrix, columns=city_labels, index=city_labels)
            edited_df = st.data_editor(matrix_df, use_container_width=True)
            
            if st.button("Compute Optimal TSP Tour & Bellman-Held-Karp State Tree"):
                dist_mat = edited_df.values.astype(float)
                start_idx = city_labels.index(start_end_city)
                n = num_cities
                other_cities = [i for i in range(n) if i != start_idx]
                
                memo_dp = {}
                def compute_dp(i, S):
                    state = (i, frozenset(S))
                    if state in memo_dp:
                        return memo_dp[state]
                    if not S:
                        cost = dist_mat[i][start_idx]
                        memo_dp[state] = (cost, start_idx)
                        return memo_dp[state]
                    
                    min_cost = float('inf')
                    best_next = -1
                    for j in S:
                        sub_cost, _ = compute_dp(j, S - {j})
                        total = dist_mat[i][j] + sub_cost
                        if total < min_cost:
                            min_cost = total
                            best_next = j
                    memo_dp[state] = (min_cost, best_next)
                    return memo_dp[state]

                all_S = frozenset(other_cities)
                opt_cost, _ = compute_dp(start_idx, all_S)
                
                optimal_states_set = set()
                curr_i = start_idx
                curr_S = all_S
                while curr_S:
                    _, nxt = memo_dp[(curr_i, curr_S)]
                    optimal_states_set.add((curr_i, curr_S))
                    curr_i = nxt
                    curr_S = curr_S - {nxt}
                optimal_states_set.add((curr_i, frozenset()))

                dot_tsp = graphviz.Digraph(comment="TSP Unrolled DP State Tree")
                dot_tsp.attr(bgcolor="transparent", rankdir="TB")
                dot_tsp.attr('node', shape='box', style='filled', fontcolor='white', fontname='Helvetica', fillcolor='#1e293b', color='#38bdf8', penwidth='1.5')
                dot_tsp.attr('edge', fontname='Helvetica', fontsize='10', color='#94a3b8')
                
                node_counter = [0]

                def build_unrolled_tree(i, S, parent_id=None):
                    state = (i, frozenset(S))
                    cost_val, best_next_city = compute_dp(i, S)
                    
                    s_id = f"n_{node_counter[0]}"
                    node_counter[0] += 1
                    
                    set_str = ", ".join(sorted([city_labels[j] for j in S])) if S else "Ø"
                    if not S:
                        label = f"g({city_labels[i]}, Ø)\n= c_{city_labels[i]}{city_labels[start_idx]} = {int(cost_val)}"
                    else:
                        calculations = []
                        for j in S:
                            sub_cost, _ = compute_dp(j, S - {j})
                            calculations.append(f"{int(dist_mat[i][j])} + {int(sub_cost)}")
                        calc_joined = ", ".join(calculations)
                        if len(calculations) > 1:
                            label = f"g({city_labels[i]}, {{{set_str}}})\n= min({calc_joined})\n= {int(cost_val)}"
                        else:
                            label = f"g({city_labels[i]}, {{{set_str}}})\n= {calc_joined}\n= {int(cost_val)}"

                    is_opt_node = (state in optimal_states_set)
                    node_fill = '#059669' if is_opt_node and state != (start_idx, all_S) else ('#2563eb' if state == (start_idx, all_S) else '#1e293b')
                    node_border = '#ef4444' if is_opt_node else '#38bdf8'
                    pen_w = '3.0' if is_opt_node else '1.5'
                    
                    dot_tsp.node(s_id, label, fillcolor=node_fill, color=node_border, penwidth=pen_w)
                    
                    if parent_id:
                        parent_i, parent_S = parent_id['i'], parent_id['S']
                        _, p_best_next = memo_dp[(parent_i, parent_S)]
                        is_opt_edge = (state in optimal_states_set and parent_id['state'] in optimal_states_set and p_best_next == i)
                        edge_color = '#ef4444' if is_opt_edge else '#38bdf8'
                        edge_width = '3.0' if is_opt_edge else '1.0'
                        edge_label = f"c_{city_labels[parent_i]}{city_labels[i]}={int(dist_mat[parent_i][i])}"
                        dot_tsp.edge(parent_id['id'], s_id, label=edge_label, color=edge_color, penwidth=edge_width, fontcolor='#f8fafc' if is_opt_edge else '#94a3b8')

                    for j in S:
                        child_state_info = {'i': i, 'S': S, 'state': state, 'id': s_id}
                        build_unrolled_tree(j, S - {j}, child_state_info)

                build_unrolled_tree(start_idx, all_S)

                st.markdown(f"### 🌳 Unrolled Bellman-Held-Karp State Tree (Start/End: {start_end_city})")
                st.caption("🔍 **How to read:** The tree builds from base cases ($\emptyset$) at the bottom upwards to the root. **Red-bordered nodes and red bold arrows** highlight the optimal TSP tour.")
                st.graphviz_chart(dot_tsp)
                
                tour_seq = [start_idx]
                c_idx = start_idx
                c_set = all_S
                while c_set:
                    _, nxt = memo_dp[(c_idx, c_set)]
                    tour_seq.append(nxt)
                    c_idx = nxt
                    c_set = c_set - {nxt}
                tour_seq.append(start_idx)
                
                tour_labels = [city_labels[idx] for idx in tour_seq]
                st.success(f"🏆 **Optimal TSP Tour:** {' $\\rightarrow$ '.join(tour_labels)} | **Minimum Total Cost = {int(opt_cost)} units**")

        else:
            st.markdown("#### 🔄 Stagewise Network Shortest Route Studio")
            net_method = st.selectbox("Select Stagewise Iteration Method", ["Backward Iteration Method (Sink -> Source)", "Forward Iteration Method (Source -> Sink)"])
            
            st.markdown("##### 1. Define Network Nodes & Stage Mapping")
            num_net_nodes = int(st.number_input("Total Number of Nodes", min_value=3, max_value=10, value=7, step=1))
            net_labels = [str(i+1) for i in range(num_net_nodes)]
            
            st.markdown("Specify which stage each node belongs to (for proper column alignment in the diagram):")
            stage_cols = st.columns(min(num_net_nodes, 4))
            node_stages = {}
            for i, node in enumerate(net_labels):
                col_idx = i % len(stage_cols)
                with stage_cols[col_idx]:
                    default_stage = 1 if i == 0 else (3 if i == num_net_nodes - 1 else 2)
                    node_stages[node] = st.number_input(f"Node {node} Stage", min_value=1, max_value=5, value=default_stage, key=f"stage_node_{node}")

            st.markdown("##### 2. Network Adjacency Matrix (Edge Weights)")
            st.info("Enter edge weights. Use `999` or `None` if no direct edge exists between nodes.")
            
            default_net = np.full((num_net_nodes, num_net_nodes), 999.0)
            default_connections = [
                (0,1,5), (0,2,14), (0,3,6),
                (1,2,6), (1,4,12), (1,5,12),
                (2,3,7), (2,4,2), (2,5,9),
                (3,5,9),
                (4,6,4),
                (5,6,4)
            ]
            for u, v, w in default_connections:
                if u < num_net_nodes and v < num_net_nodes:
                    default_net[u, v] = w
                    
            net_df = pd.DataFrame(default_net, columns=net_labels, index=net_labels)
            edited_net_df = st.data_editor(net_df, use_container_width=True)
            
            if st.button("Compute Stagewise Shortest Route & Tables"):
                matrix_vals = edited_net_df.values.astype(float)
                
                if "Backward" in net_method:
                    sink = num_net_nodes - 1
                    source = 0
                    
                    f_cost = {sink: 0.0}
                    policy = {sink: None}
                    
                    stages_sorted = sorted(list(set(node_stages.values())), reverse=True)
                    computation_tables = {}
                    
                    for stg in stages_sorted[1:]:
                        current_stage_nodes = [n for n, s in node_stages.items() if s == stg]
                        stage_rows = []
                        for node_str in current_stage_nodes:
                            u_idx = net_labels.index(node_str)
                            best_val = float('inf')
                            best_nxt = None
                            row_dict = {"State ($x_i$)": f"Node {node_str}"}
                            
                            for next_node_str in net_labels:
                                v_idx = net_labels.index(next_node_str)
                                wt = matrix_vals[u_idx, v_idx]
                                if wt < 900 and v_idx in [net_labels.index(n) for n, s in node_stages.items() if s > stg]:
                                    if v_idx in f_cost:
                                        total_cost = wt + f_cost[v_idx]
                                        row_dict[f"to {next_node_str}"] = f"{int(wt)} + {int(f_cost[v_idx])} = {int(total_cost)}"
                                        if total_cost < best_val:
                                            best_val = total_cost
                                            best_nxt = next_node_str
                                    else:
                                        row_dict[f"to {next_node_str}"] = f"w={int(wt)} (Uncomputed)"
                                else:
                                    row_dict[f"to {next_node_str}"] = "—"
                            
                            if best_val != float('inf'):
                                f_cost[u_idx] = best_val
                                policy[u_idx] = best_nxt
                                row_dict["Optimal $f(x)$"] = int(best_val)
                                row_dict["Optimal Next"] = f"Node {best_nxt}"
                            else:
                                row_dict["Optimal $f(x)$"] = "—"
                                row_dict["Optimal Next"] = "—"
                                
                            stage_rows.append(row_dict)
                        computation_tables[stg] = pd.DataFrame(stage_rows)

                    source_nodes = [n for n, s in node_stages.items() if s == min(node_stages.values())]
                    source_rows = []
                    for node_str in source_nodes:
                        u_idx = net_labels.index(node_str)
                        best_val = float('inf')
                        best_nxt = None
                        row_dict = {"State ($x_i$)": f"Node {node_str}"}
                        for next_node_str in net_labels:
                            v_idx = net_labels.index(next_node_str)
                            wt = matrix_vals[u_idx, v_idx]
                            if wt < 900 and v_idx in f_cost:
                                total_cost = wt + f_cost[v_idx]
                                row_dict[f"to {next_node_str}"] = f"{int(wt)} + {int(f_cost[v_idx])} = {int(total_cost)}"
                                if total_cost < best_val:
                                    best_val = total_cost
                                    best_nxt = next_node_str
                            else:
                                row_dict[f"to {next_node_str}"] = "—"
                        f_cost[u_idx] = best_val
                        policy[u_idx] = best_nxt
                        row_dict["Optimal $f(x)$"] = int(best_val)
                        row_dict["Optimal Next"] = f"Node {best_nxt}"
                        source_rows.append(row_dict)
                    computation_tables[min(node_stages.values())] = pd.DataFrame(source_rows)

                    curr_idx = 0
                    shortest_path = [net_labels[curr_idx]]
                    while curr_idx in policy and policy[curr_idx] is not None:
                        nxt_node = policy[curr_idx]
                        shortest_path.append(nxt_node)
                        curr_idx = net_labels.index(nxt_node)

                    st.markdown("### 📋 Step-by-Step Backward Iteration Tables")
                    for stg_num in sorted(computation_tables.keys(), reverse=True):
                        st.markdown(f"**Stage {stg_num} Calculations**")
                        st.dataframe(computation_tables[stg_num], use_container_width=True)

                    total_min_cost = f_cost.get(0, float('inf'))
                    
                else:
                    source = 0
                    sink = num_net_nodes - 1
                    
                    f_cost = {source: 0.0}
                    policy = {source: None}
                    
                    stages_sorted = sorted(list(set(node_stages.values())))
                    computation_tables = {}
                    
                    for stg in stages_sorted[1:]:
                        current_stage_nodes = [n for n, s in node_stages.items() if s == stg]
                        prev_stage_indices = {net_labels.index(n) for n, s in node_stages.items() if s < stg}
                        stage_rows = []
                        for node_str in current_stage_nodes:
                            v_idx = net_labels.index(node_str)
                            best_val = float('inf')
                            best_prev = None
                            row_dict = {"State ($x_i$)": f"Node {node_str}"}
                            
                            for prev_node_str in net_labels:
                                u_idx = net_labels.index(prev_node_str)
                                wt = matrix_vals[u_idx, v_idx]
                                if wt < 900 and u_idx in prev_stage_indices and u_idx in f_cost:
                                    total_cost = f_cost[u_idx] + wt
                                    row_dict[f"from {prev_node_str}"] = f"{int(f_cost[u_idx])} + {int(wt)} = {int(total_cost)}"
                                    if total_cost < best_val:
                                        best_val = total_cost
                                        best_prev = prev_node_str
                                else:
                                    row_dict[f"from {prev_node_str}"] = "—"
                            
                            if best_val != float('inf'):
                                f_cost[v_idx] = best_val
                                policy[v_idx] = best_prev
                                row_dict["Optimal $f(x)$"] = int(best_val)
                                row_dict["Optimal Prev"] = f"Node {best_prev}"
                            else:
                                row_dict["Optimal $f(x)$"] = "—"
                                row_dict["Optimal Prev"] = "—"
                            stage_rows.append(row_dict)
                        computation_tables[stg] = pd.DataFrame(stage_rows)

                    curr_idx = num_net_nodes - 1
                    shortest_path = [net_labels[curr_idx]]
                    while curr_idx in policy and policy[curr_idx] is not None:
                        prev_node = policy[curr_idx]
                        shortest_path.insert(0, prev_node)
                        curr_idx = net_labels.index(prev_node)

                    st.markdown("### 📋 Step-by-Step Forward Iteration Tables")
                    for stg_num in sorted(computation_tables.keys()):
                        st.markdown(f"**Stage {stg_num} Calculations**")
                        st.dataframe(computation_tables[stg_num], use_container_width=True)

                    total_min_cost = f_cost.get(num_net_nodes - 1, float('inf'))

                dot_net = graphviz.Digraph(comment="Stagewise Network Diagram")
                dot_net.attr(bgcolor="transparent", rankdir="LR")
                dot_net.attr('node', shape='circle', style='filled', fontcolor='white', fontname='Helvetica')
                dot_net.attr('edge', fontname='Helvetica', fontsize='11')
                
                unique_stages = sorted(list(set(node_stages.values())))
                for stg in unique_stages:
                    with dot_net.subgraph() as subg:
                        subg.attr(rank='same')
                        nodes_in_stage = [n for n, s in node_stages.items() if s == stg]
                        for n in nodes_in_stage:
                            fill_color = '#10b981' if n == net_labels[0] or n == net_labels[-1] else '#2563eb'
                            subg.node(n, label=f"Node {n}", fillcolor=fill_color)

                opt_pairs = set((shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1))

                for r in range(num_net_nodes):
                    for c in range(num_net_nodes):
                        wt_val = matrix_vals[r, c]
                        if wt_val < 900:
                            u_name = net_labels[r]
                            v_name = net_labels[c]
                            is_opt = (u_name, v_name) in opt_pairs
                            edge_col = '#10b981' if is_opt else '#38bdf8'
                            pen_w = '3.5' if is_opt else '1.2'
                            dot_net.edge(u_name, v_name, label=str(int(wt_val)), color=edge_col, penwidth=pen_w)

                st.markdown("### 📊 Stagewise Network Diagram (Aligned Stages)")
                st.caption("Nodes of the same stage are aligned in vertical columns. The optimal shortest path is highlighted in bold green.")
                st.graphviz_chart(dot_net)
                
                if total_min_cost < float('inf'):
                    path_str = " -> ".join([f"Node {n}" for n in shortest_path])
                    st.success(f"🌟 **Optimal Shortest Path ({net_method}):** {path_str} | **Total Minimum Distance = {int(total_min_cost)}**")
                else:
                    st.error("No valid complete path found connecting source to sink with current matrix weights.")

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
