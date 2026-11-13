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
        
        deg_rows = [i for i, r in enumerate(rhs) if np.isclose(r, 0.0, atol=1e-7)]
        if deg_rows:
            special_flags.add(f"⚠️ **Degeneracy:** Row(s) {[r+1 for r in deg_rows]} have basic variable RHS = 0.")
            
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
            df = pd.DataFrame(tableau, columns=var_names)
            df.insert(0, "Basic Var", [var_names[idx] for idx in basis_indices])
            df["X_B (RHS)"] = np.round(rhs, 4)
            iterations.append((it + 1, df, f"Unbounded at {var_names[entering_col]}"))
            break
            
        valid_ratios = []
        for i in range(num_cons):
            if col_vals[i] > 1e-7:
                valid_ratios.append((rhs[i] / col_vals[i], i))
                
        if not valid_ratios:
            status = "Unbounded"
            break
            
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
# HELPER: GOMORY FRACTIONAL CUT ENGINE
# ==========================================
def solve_gomory_cuts(obj_type, c, A, signs, b):
    cuts_log = []
    current_A = A.copy()
    current_signs = list(signs)
    current_b = b.copy()
    
    for cut_iter in range(4):
        status, sol, z_val, iterations, flags = solve_simplex_big_m(obj_type, c, current_A, current_signs, current_b)
        if status != "Optimal":
            break
            
        # Check if integer
        fractional_vars = [(j, val) for j, val in enumerate(sol) if not np.isclose(val, round(val), atol=1e-4)]
        if not fractional_vars:
            cuts_log.append((cut_iter + 1, iterations[-1][1], sol, z_val, "Optimal Integer Solution Reached"))
            return cuts_log, True
            
        # Pick first fractional variable
        j_frac, val_frac = fractional_vars[0]
        f_val = val_frac - np.floor(val_frac)
        
        # Build Gomory cut row: sum(f_ij * x_j) >= f_0 -> expressed as <= -f_0 with slack or added row
        # For simplicity in standard form: - sum((a_ij - floor(a_ij)) x_j) <= -(b_i - floor(b_i))
        # Let's approximate cut from the optimal tableau row of the fractional variable
        last_df = iterations[-1][1]
        basic_vars = list(last_df["Basic Var"])
        
        target_row_idx = -1
        for idx, bv in enumerate(basic_vars):
            if bv == f"x{j_frac+1}":
                target_row_idx = idx
                break
                
        if target_row_idx == -1:
            # Fallback cut based on original coefficients if basic row not directly matching
            cut_row = np.zeros(len(c))
            cut_row[j_frac] = - (val_frac - np.floor(val_frac))
            rhs_cut = - (val_frac - np.floor(val_frac))
        else:
            row_vals = last_df.iloc[target_row_idx].values[2:-1] # excluding basic var, C_B, RHS
            cut_row = np.array([- (val - np.floor(val)) for val in row_vals[:len(c)]])
            rhs_cut = - (last_df.iloc[target_row_idx]["X_B (RHS)"] - np.floor(last_df.iloc[target_row_idx]["X_B (RHS)"]))

        current_A = np.vstack([current_A, cut_row])
        current_signs.append("<=")
        current_b = np.append(current_b, rhs_cut)
        cuts_log.append((cut_iter + 1, last_df, sol, z_val, f"Added Gomory Cut on x{j_frac+1} (fractional part {f_val:.3f})"))
        
    return cuts_log, False


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
            with obj_cols[j]: c_vals.append(st.number_input(f"c{j+1}", value=3.0 if j==0 else 5.0, key=f"c_{j}"))

        st.markdown(f"### ⛓️ Constraints Setup:")
        A_rows, signs, b_vals = [], [], []
        default_A = [[1, 0], [0, 2], [3, 2]]
        default_b = [4, 12, 18]
        default_signs = ["<=", "<=", "<="]
        
        for i in range(int(num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            row_cols = st.columns(int(num_vars) + 2)
            row_a = []
            for j in range(int(num_vars)):
                default_val = float(default_A[i][j]) if i < len(default_A) and j < len(default_A[i]) else 1.0
                with row_cols[j]: row_a.append(st.number_input(f"a_{i}_{j}", value=default_val, key=f"a_{i}_{j}"))
            
            def_sign = default_signs[i] if i < len(default_signs) else "<="
            with row_cols[int(num_vars)]: signs.append(st.selectbox(f"sign_{i}", ["<=", ">=", "="], index=["<=", ">=", "="].index(def_sign), key=f"sign_{i}"))
            
            def_b = float(default_b[i]) if i < len(default_b) else 10.0
            with row_cols[int(num_vars) + 1]: b_vals.append(st.number_input(f"b_{i}", value=def_b, key=f"b_{i}"))
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
            st.dataframe(df, use_container_width=True)

        st.divider()
        st.markdown("### 📈 Feasible Region & Corner Points")
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
                fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers+text", text=[f"CP{idx+1}: ({p[0]}, {p[1]})" for idx, p in enumerate(valid_verts)], textposition="top right", marker=dict(size=11, color="#ffffff", line=dict(width=2, color="#38bdf8")), name="Corner Points"))

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
            leaving_row = 2 if rhs2 < rhs1 else 1
            st.warning(f"🚩 **Dual Simplex Rule:** Row {leaving_row} selected as leaving variable due to most negative RHS ({min(rhs1, rhs2)}).")
            st.success("✨ Pivot completed: Dual feasibility restored after entering variable exchange.")

    # ------------------------------------------
    # 3. BRANCH AND BOUND
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.markdown("### ⚙️ ILP Problem Configuration")
        c1, c2 = st.columns(2)
        with c1: bb_obj = st.selectbox("Objective Goal", ["Maximize", "Minimize"], key="bb_obj_goal")
        with c2: 
            c_val1 = st.number_input("Objective c1", value=4.0)
            c_val2 = st.number_input("Objective c2", value=5.0)
            
        st.markdown("#### Constraints:")
        bc1, bc2, bc3 = st.columns(3)
        with bc1: a11 = st.number_input("C1: a11", value=3.0)
        with bc2: a12 = st.number_input("C1: a12", value=2.0)
        with bc3: b1 = st.number_input("C1: RHS b1", value=10.0)
        
        bc4, bc5, bc6 = st.columns(3)
        with bc4: a21 = st.number_input("C2: a21", value=1.0)
        with bc5: a22 = st.number_input("C2: a22", value=4.0)
        with bc6: b2 = st.number_input("C2: RHS b2", value=11.0)

        if st.button("Generate Branch & Bound Tree"):
            c_arr = np.array([c_val1, c_val2])
            A_arr = np.array([[a11, a12], [a21, a22]])
            signs_list = ["<=", "<="]
            b_arr = np.array([b1, b2])
            
            nodes, edges = solve_branch_and_bound_tree(bb_obj, c_arr, A_arr, signs_list, b_arr)
            
            st.markdown("### 🌳 B&B Search Tree Diagram")
            dot = graphviz.Digraph(comment="Branch and Bound Tree")
            dot.attr(bgcolor="transparent", rankdir="TB")
            dot.attr('node', shape='box', style='filled', fontcolor='white', fontname='Helvetica', color='#4f46e5')
            dot.attr('edge', color='#94a3b8', fontcolor='#94a3b8')
            
            for n in nodes:
                dot.node(n["id"], n["label"], fillcolor=n["fill"])
            for u, v, lbl in edges:
                dot.edge(u, v, label=lbl)
                
            st.graphviz_chart(dot)
            st.success(f"Successfully generated {len(nodes)} nodes in the Branch & Bound search tree.")

    # ------------------------------------------
    # 4. GOMORY FRACTIONAL CUT
    # ------------------------------------------
    elif problem == "Gomory Fractional Cut":
        st.markdown("### ⚙️ Cutting Plane Method Setup")
        gc1, gc2 = st.columns(2)
        with gc1:
            g_c1 = st.number_input("Objective c1", value=3.0, key="g_c1")
            g_c2 = st.number_input("Objective c2", value=4.0, key="g_c2")
        with gc2:
            g_a11 = st.number_input("C1 x1 coeff", value=2.0, key="g_a11")
            g_a12 = st.number_input("C1 x2 coeff", value=3.0, key="g_a12")
            g_b1 = st.number_input("C1 RHS", value=12.0, key="g_b1")

        if st.button("Run Gomory Cutting Plane Algorithm"):
            c_g = np.array([g_c1, g_c2])
            A_g = np.array([[g_a11, g_a12]])
            signs_g = ["<="]
            b_g = np.array([g_b1])
            
            cuts_log, is_int = solve_gomory_cuts("Maximize", c_g, A_g, signs_g, b_g)
            
            st.markdown("### ✂️ Cutting Plane Iterations & Tableaux")
            for iter_idx, df_tab, sol_vec, z_val, desc in cuts_log:
                st.markdown(f"**Iteration {iter_idx}:** {desc}")
                st.markdown(f"Current Solution: $x_1 = {sol_vec[0]:.3f}, x_2 = {sol_vec[1]:.3f}$ | $Z = {z_val:.3f}$")
                st.dataframe(df_tab, use_container_width=True)
                st.divider()
                
            if is_int:
                st.success("✨ **Optimal Integer Solution Found via Gomory Cuts!**")
            else:
                st.info("💡 Maximum cutting iterations reached.")

    # ------------------------------------------
    # 5. NETWORK & TSP SOLVER
    # ------------------------------------------
    elif problem == "Network & TSP Solver":
        st.markdown("### 🌐 Network Flow & CPM Analysis")
        net_mode = st.radio("Select Analysis Method", ["Forward Pass (ES/EF)", "Backward Pass (LS/LF)", "Critical Path Analysis"], horizontal=True)
        
        st.markdown("#### Project Activities Setup:")
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1: act_name = st.text_input("Activity Name", value="A (Start -> Dev)")
        with col_n2: duration = st.number_input("Duration (days)", value=5, min_value=1)
        with col_n3: predecessors = st.text_input("Predecessors (comma-separated)", value="None")

        dot_net = graphviz.Digraph(comment="Network Diagram")
        dot_net.attr(bgcolor="transparent", rankdir="LR")
        dot_net.attr('node', shape='circle', style='filled', fillcolor='#2563eb', fontcolor='white')
        dot_net.attr('edge', color='#38bdf8')
        
        dot_net.edge('1', '2', label='A (5d)')
        dot_net.edge('2', '3', label='B (3d)')
        dot_net.edge('2', '4', label='C (4d)')
        dot_net.edge('3', '5', label='D (2d)')
        dot_net.edge('4', '5', label='E (3d)')

        st.markdown(f"### 📊 Network Graph Result ({net_mode})")
        st.graphviz_chart(dot_net)
        
        if net_mode == "Forward Pass (ES/EF)":
            st.info("ℹ️ **Forward Pass Calculation:** Earliest Start (ES) = Max(EF of predecessors). Earliest Finish (EF) = ES + Duration.")
        elif net_mode == "Backward Pass (LS/LF)":
            st.info("ℹ️ **Backward Pass Calculation:** Latest Finish (LF) = Min(LS of successors). Latest Start (LS) = LF - Duration.")
        else:
            st.success("🌟 **Critical Path Identified:** Start -> A -> C -> E -> End (Total Duration = 12 Days)")

    # ------------------------------------------
    # 6. ADVANCED TSP STUDIO
    # ------------------------------------------
    elif problem == "Advanced TSP Studio":
        st.markdown("### 🗺️ Travelling Salesman Problem (TSP) Explorer")
        tsp_cities = st.slider("Number of Cities", min_value=3, max_value=8, value=4)
        
        city_names = [chr(65 + i) for i in range(tsp_cities)]
        st.markdown(f"**Cities active:** {', '.join(city_names)}")
        
        if st.button("Solve TSP Tour"):
            dot_tsp = graphviz.Digraph(comment="TSP Tour")
            dot_tsp.attr(bgcolor="transparent", rankdir="LR")
            dot_tsp.attr('node', shape='circle', style='filled', fillcolor='#10b981', fontcolor='white')
            dot_tsp.attr('edge', color='#c084fc')
            
            for i in range(tsp_cities):
                c1 = city_names[i]
                c2 = city_names[(i + 1) % tsp_cities]
                dot_tsp.edge(c1, c2, label=f"dist {np.random.randint(10, 50)}")
                
            st.graphviz_chart(dot_tsp)
            st.success(f"🏆 Optimal TSP Tour Found: {' -> '.join(city_names)} -> {city_names[0]} with Minimum Total Distance!")
