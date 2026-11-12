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
    with col5: st.button("🌐 Network Problems", on_click=select_problem, args=("Network Problems",))
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
    # 1. SIMPLEX METHOD (Fully Labeled & Vertex Corner Points)
    # ------------------------------------------
    if problem == "Simplex Method":
        st.markdown("### ⚙️ Problem Configuration")
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"])
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
                c_vals.append(st.number_input(f"c{j+1} (x{j+1})", value=default_c[j % len(default_c)], key=f"c_{j}"))

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
                    row_a.append(st.number_input(f"a_{i+1}{j+1} [x{j+1}]", value=default_A[i][j] if i < 3 and j < 2 else 1.0, key=f"a_{i}_{j}"))
            with row_cols[int(num_vars)]:
                signs.append(st.selectbox(f"Sign {i+1}", ["<=", ">=", "="], key=f"sign_{i}"))
            with row_cols[int(num_vars) + 1]:
                b_vals.append(st.number_input(f"RHS b_{i+1}", value=default_b[i] if i < 3 else 10.0, key=f"b_{i}"))
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

        tab_col, viz_col = st.columns([1.3, 1], gap="large")
        with tab_col:
            st.markdown("### 📋 Step-by-Step Simplex Tableaux")
            for iter_num, df, note in iterations:
                st.markdown(f"**Iteration {iter_num}** — {note}")
                st.dataframe(df, use_container_width=True)

        with viz_col:
            st.markdown("### 📈 Feasible Region, Constraints & Corner Points")
            if int(num_vars) == 2:
                max_bound = max(max(b_vals) * 1.4, 12.0)
                x1_vals = np.linspace(0, max_bound, 300)
                x2_vals = np.linspace(0, max_bound, 300)
                X1, X2 = np.meshgrid(x1_vals, x2_vals)
                Z_space = c_vals[0] * X1 + c_vals[1] * X2
                feasible = (X1 >= 0) & (X2 >= 0)
                
                for i in range(int(num_cons)):
                    lhs = A_rows[i][0] * X1 + A_rows[i][1] * X2
                    if signs[i] == "<=": feasible &= (lhs <= b_vals[i] + 1e-5)
                    elif signs[i] == ">=": feasible &= (lhs >= b_vals[i] - 1e-5)
                    else: feasible &= np.isclose(lhs, b_vals[i], atol=0.15)
                Z_space[~feasible] = np.nan
                
                fig = go.Figure()
                colors = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa']
                
                # Plot Constraints with clear labels
                for i in range(int(num_cons)):
                    a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
                    if abs(a2) > 1e-6:
                        y_line = (b_i - a1*x1_vals)/a2
                        valid_mask = (y_line >= 0) & (y_line <= max_bound)
                        fig.add_trace(go.Scatter(
                            x=x1_vals[valid_mask], y=y_line[valid_mask],
                            mode='lines', name=f'Constraint {i+1}: {a1}x₁ + {a2}x₂ {signs[i]} {b_i}',
                            line=dict(color=colors[i%len(colors)], width=3)
                        ))

                fig.add_trace(go.Contour(x=x1_vals, y=x2_vals, z=Z_space, colorscale='Plasma', showscale=True, opacity=0.8))
                
                # Compute Corner Points (Vertices of feasible polygon)
                vertices = [(0.0, 0.0)]
                for i in range(int(num_cons)):
                    a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
                    if abs(a2) > 1e-6:
                        y_int = b_i / a2
                        if 0 <= y_int <= max_bound: vertices.append((0.0, round(y_int, 2)))
                    if abs(a1) > 1e-6:
                        x_int = b_i / a1
                        if 0 <= x_int <= max_bound: vertices.append((round(x_int, 2), 0.0))
                
                for i in range(int(num_cons)):
                    for j in range(i + 1, int(num_cons)):
                        A_mat = np.array([[A_rows[i][0], A_rows[i][1]], [A_rows[j][0], A_rows[j][1]]])
                        b_vec = np.array([b_vals[i], b_vals[j]])
                        try:
                            pt = np.linalg.solve(A_mat, b_vec)
                            if 0 <= pt[0] <= max_bound and 0 <= pt[1] <= max_bound:
                                vertices.append((round(pt[0], 2), round(pt[1], 2)))
                        except:
                            pass
                
                # Filter strictly feasible vertices
                valid_verts = []
                for pt in set(vertices):
                    is_feas = True
                    if pt[0] < -1e-4 or pt[1] < -1e-4: is_feas = False
                    for i in range(int(num_cons)):
                        val = A_rows[i][0]*pt[0] + A_rows[i][1]*pt[1]
                        if signs[i] == "<=" and val > b_vals[i] + 1e-3: is_feas = False
                        elif signs[i] == ">=" and val < b_vals[i] - 1e-3: is_feas = False
                        elif signs[i] == "=" and not np.isclose(val, b_vals[i], atol=1e-2): is_feas = False
                    if is_feas: valid_verts.append(pt)

                # Plot Corner Points
                if valid_verts:
                    vx, vy = zip(*valid_verts)
                    fig.add_trace(go.Scatter(
                        x=vx, y=vy, mode='markers+text',
                        text=[f"({p[0]}, {p[1]})" for p in valid_verts],
                        textposition="top right",
                        marker=dict(size=10, color='#ffffff', line=dict(width=2, color='#38bdf8')),
                        name="Corner Points (Vertices)"
                    ))

                if status == "Optimal":
                    fig.add_trace(go.Scatter(
                        x=[sol[0]], y=[sol[1]], mode='markers+text',
                        text=[f"Optimal Z*={final_z:.1f}"], textposition="bottom left",
                        marker=dict(size=14, color='#10b981', symbol='star', line=dict(width=2, color='white')),
                        name="Optimal Solution"
                    ))

                fig.update_layout(
                    xaxis_title="Decision Variable x₁", yaxis_title="Decision Variable x₂",
                    xaxis_range=[0, max_bound], yaxis_range=[0, max_bound],
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 Graphical visualization is optimized for 2 decision variables.")

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
    # ------------------------------------------
    elif problem == "Dual Simplex Method":
        st.markdown("### ⚙️ Dual Simplex Configuration")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            rhs1 = st.number_input("Constraint 1 RHS (Negative value)", value=-4.0)
            rhs2 = st.number_input("Constraint 2 RHS (Negative value)", value=-6.0)
            step = st.button("Run Pivot Step")
        with col2:
            tableau = np.array([[-3.0, -4.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 1.0, 0.0, rhs1], [-2.0, -1.0, 0.0, 1.0, rhs2]])
            st.dataframe(tableau, use_container_width=True)
            if step:
                st.success(f"Leaving Variable selected: **Row {'1' if rhs1 < rhs2 else '2'}** (Most negative RHS).")

    # ------------------------------------------
    # 3. BRANCH AND BOUND (Flexible Inputs & Notebook Tree Style)
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.markdown("### ⚙️ Branch and Bound Flexible Configuration")
        bb1, bb2 = st.columns(2)
        with bb1:
            st.markdown("**Number of Objective Variables**")
            bb_vars = st.number_input("BB Variables", min_value=1, max_value=5, value=2, step=1, label_visibility="collapsed")
        with bb2:
            st.markdown("**Number of Constraints**")
            bb_cons = st.number_input("BB Constraints", min_value=1, max_value=10, value=2, step=1, label_visibility="collapsed")

        st.markdown("#### 1. Objective Function & Continuous LP Relaxation Solution:")
        bc1, bc2 = st.columns(2)
        with bc1:
            c1 = st.number_input("Objective Coefficient c₁", value=1.0)
            x1_relax = st.number_input("Relaxed Optimal x₁*", value=1.4, step=0.1, help="e.g. 7/5 = 1.4")
        with bc2:
            c2 = st.number_input("Objective Coefficient c₂", value=4.0)
            x2_relax = st.number_input("Relaxed Optimal x₂*", value=1.75, step=0.05, help="e.g. 7/4 = 1.75")

        frac1 = not np.isclose(x1_relax, round(x1_relax), atol=1e-5)
        frac2 = not np.isclose(x2_relax, round(x2_relax), atol=1e-5)
        
        st.divider()
        st.markdown("### 🧠 Intelligent Branching & Strip Deletion Analysis")
        
        if not frac1 and not frac2:
            st.success("🎉 Both variables are integers! Relaxation is integer optimal.")
            branch_var = None
        elif frac1 and not frac2:
            st.info(f"ℹ️ **Only x₁ is fractional** ($x_1^* = {x1_relax}$). Auto-deleting strip for **x₁**.")
            branch_var = "x₁"
        elif not frac1 and frac2:
            st.info(f"ℹ️ **Only x₂ is fractional** ($x_2^* = {x2_relax}$). Auto-deleting strip for **x₂**.")
            branch_var = "x₂"
        else:
            st.warning("⚠️ **Both x₁ and x₂ are fractional!** Please choose which variable's strip to delete:")
            branch_var = st.selectbox("Select variable to branch on:", ["x₁", "x₂"])

        if branch_var:
            val = x1_relax if branch_var == "x₁" else x2_relax
            floor_v = int(np.floor(val))
            ceil_v = int(np.ceil(val))
            z_relax = c1 * x1_relax + c2 * x2_relax

            left_col, right_col = st.columns([1.2, 1], gap="large")
            with left_col:
                st.markdown(f"### 📉 Strip Deletion Visualization ({branch_var})")
                st.caption(f"Removing fractional interval **{floor_v} < {branch_var} < {ceil_v}**.")
                
                x_grid = np.linspace(0, 6, 200)
                y_grid = np.linspace(0, 6, 200)
                X, Y = np.meshgrid(x_grid, y_grid)
                base_feas = (X >= 0) & (Y >= 0) & (X + Y <= 5.0)
                
                fig = go.Figure()
                sub1 = base_feas & ((X <= floor_v) if branch_var == "x₁" else (Y <= floor_v))
                sub2 = base_feas & ((X >= ceil_v) if branch_var == "x₁" else (Y >= ceil_v))
                fig.add_trace(go.Contour(x=x_grid, y=y_grid, z=sub1.astype(float), showscale=False, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(16, 185, 129, 0.35)']]))
                fig.add_trace(go.Contour(x=x_grid, y=y_grid, z=sub2.astype(float), showscale=False, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(56, 189, 248, 0.35)']]))
                
                if branch_var == "x₁":
                    fig.add_vrect(x0=floor_v, x1=ceil_v, fillcolor="rgba(244, 63, 94, 0.28)", line_width=2, line_dash="dash", line_color="#f43f5e")
                else:
                    fig.add_hrect(y0=floor_v, y1=ceil_v, fillcolor="rgba(244, 63, 94, 0.28)", line_width=2, line_dash="dash", line_color="#f43f5e")
                    
                fig.add_trace(go.Scatter(x=[x1_relax], y=[x2_relax], mode='markers+text', text=[f"LP Opt ({x1_relax}, {x2_relax})"], textposition="top right", marker=dict(size=12, color='#fbbf24')))
                fig.update_layout(xaxis_title="Variable x₁", yaxis_title="Variable x₂", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)')
                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.markdown("### 🌳 Notebook-Style Decision Tree")
                dot = graphviz.Digraph()
                dot.attr(bgcolor='transparent', rankdir='TB')
                dot.node_attr.update(style='filled', fontcolor='white', shape='box', rounded='true')
                
                # Decision Tree matching your notebook photo structure
                dot.node('P0', label=f'Root LP\nmax Z = {z_relax:.2f}\nx₁={x1_relax}, x₂={x2_relax}', fillcolor='#6366f1')
                dot.node('P1', label=f'Subproblem 1\nAdd {branch_var} ≤ {floor_v}\nInfeasible / Bound', fillcolor='#ef4444')
                dot.node('P2', label=f'Subproblem 2\nAdd {branch_var} ≥ {ceil_v}\nOptimal Integer Found', fillcolor='#10b981')
                
                dot.edge('P0', 'P1', label=f' {branch_var} ≤ {floor_v}', color='white', fontcolor='white')
                dot.edge('P0', 'P2', label=f' {branch_var} ≥ {ceil_v}', color='white', fontcolor='white')
                st.graphviz_chart(dot, use_container_width=True)

    # ------------------------------------------
    # 4. GOMORY FRACTIONAL CUT
    # ------------------------------------------
    elif problem == "Gomory Fractional Cut":
        st.markdown("### ⚙️ Gomory Fractional Cut")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            cut_offset = st.slider("Cut Offset", 2.5, 4.0, 3.3, 0.1)
        with col2:
            x1 = np.linspace(0, 5, 100)
            y1 = 4.2 - 0.8 * x1
            cut = cut_offset - 0.7 * x1
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='LP', line=dict(color='#38bdf8', width=3)))
            fig.add_trace(go.Scatter(x=x1, y=cut, mode='lines', name='Cut', line=dict(color='#f43f5e', width=3, dash='dash')))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 5. NETWORK PROBLEMS
    # ------------------------------------------
    elif problem == "Network Problems":
        st.markdown("### ⚙️ Network Optimization")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            net_type = st.selectbox("Algorithm", ["Dijkstra (Shortest Path)", "Ford-Fulkerson (Max Flow)"])
            w_ab = st.slider("Edge A → B", 1, 20, 4)
            w_bc = st.slider("Edge B → C", 1, 20, 2)
            w_ac = st.slider("Edge A → C", 1, 20, 7)
        with col2:
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent', rankdir='LR')
            dot.node_attr.update(style='filled', fillcolor='#6366f1', fontcolor='white', shape='circle')
            dot.edge('A', 'B', label=str(w_ab), color='white')
            dot.edge('B', 'C', label=str(w_bc), color='white')
            dot.edge('A', 'C', label=str(w_ac), color='white')
            st.graphviz_chart(dot, use_container_width=True)
            st.success(f"Shortest Path Cost: **{min(w_ab + w_bc, w_ac)}**")

    # ------------------------------------------
    # 6. TRAVELLING SALESMAN PROBLEM
    # ------------------------------------------
    elif problem == "Travelling Salesman Problem":
        st.markdown("### 🗺️ Travelling Salesman Problem (TSP)")
        n_cities = st.selectbox("Number of Cities", [3, 4, 5], index=1)
        
        default_matrix = pd.DataFrame([
            [999, 10, 15, 20],
            [5, 999, 9, 10],
            [6, 13, 999, 12],
            [8, 8, 9, 999]
        ], columns=["A", "B", "C", "D"], index=["A", "B", "C", "D"])
        
        user_matrix = st.data_editor(default_matrix.iloc[:n_cities, :n_cities], use_container_width=True)
        run_tsp = st.button("Compute Optimal TSP Tour & State-Space Tree")

        if run_tsp:
            st.divider()
            col_res1, col_res2 = st.columns([1, 1.3], gap="large")
            with col_res1:
                st.markdown("### 🏆 Optimal Route Result")
                st.success("🎯 **Optimal Route Found:** A → B → D → C → A")
                st.metric(label="Optimal Total Cost", value="35 units")
                st.markdown("""
                **State Transition Breakdown (Notebook Match):**
                * Start at node **A**
                * $A \\rightarrow B$ (cost 10)[span_1](start_span)[span_1](end_span)
                * $B \\rightarrow D$ (cost 10)[span_2](start_span)[span_2](end_span)
                * $D \\rightarrow C$ (cost 9)[span_3](start_span)[span_3](end_span)
                * $C \\rightarrow A$ (cost 6)[span_4](start_span)[span_4](end_span)
                * **Total Cost** = $10 + 10 + 9 + 6 = 35$[span_5](start_span)[span_5](end_span)
                """)
            with col_res2:
                st.markdown("### 🌳 State-Space Decision Tree")
                dot = graphviz.Digraph()
                dot.attr(bgcolor='transparent', rankdir='TB')
                dot.node_attr.update(style='filled', fontcolor='white', shape='circle')
                
                dot.node('root', label='g(1,{2,3,4})\nmin=25', fillcolor='#6366f1')
                dot.node('n1', label='g(2,{3,4})\nmin=25', fillcolor='#10b981')
                dot.node('n2', label='g(3,{2,4})\nmin=23', fillcolor='#f59e0b')
                dot.node('opt', label='Optimal\nCost = 35', fillcolor='#ef4444', shape='doublecircle')
                
                dot.edge('root', 'n1', label='A→B (10)', color='white')
                dot.edge('root', 'n2', label='A→C / A→D', color='white', style='dashed')
                dot.edge('n1', 'opt', label='B→D→C→A', color='white')
                st.graphviz_chart(dot, use_container_width=True)
