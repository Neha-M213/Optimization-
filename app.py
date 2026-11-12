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
# HELPER: FULL BIG-M SIMPLEX ENGINE
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
            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    c_eff = c.copy().astype(float) if obj_type == "Maximize" else -c.copy().astype(float)
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)
    
    extra_cols, basis_indices, artificial_indices = [], [], []
    s_count, a_count = 1, 1
    for i, s in enumerate(signs):
        if s == "<=":
            col = np.zeros(num_cons)
            col[i] = 1.0
            extra_cols.append(col)
            var_names.append(f"s{s_count}")
            cost_row.append(0.0)
            basis_indices.append(len(var_names) - 1)
            s_count += 1
        elif s == ">=":
            col_s = np.zeros(num_cons)
            col_s[i] = -1.0
            extra_cols.append(col_s)
            var_names.append(f"s{s_count}")
            cost_row.append(0.0)
            s_count += 1
            
            col_a = np.zeros(num_cons)
            col_a[i] = 1.0
            extra_cols.append(col_a)
            var_names.append(f"A{a_count}")
            cost_row.append(-M)
            basis_indices.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)
            a_count += 1
        elif s == "=":
            col_a = np.zeros(num_cons)
            col_a[i] = 1.0
            extra_cols.append(col_a)
            var_names.append(f"A{a_count}")
            cost_row.append(-M)
            basis_indices.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)
            a_count += 1

    tableau = np.hstack([A, np.column_stack(extra_cols)]) if extra_cols else A.copy()
    cost_row = np.array(cost_row)
    rhs = b.copy()
    
    iterations, special_flags = [], set()
    status = "Optimal"
    
    for it in range(25):
        cb = cost_row[basis_indices]
        zj = cb @ tableau
        cj_zj = cost_row - zj
        z_val = cb @ rhs
        
        if any(np.isclose(rhs, 0.0, atol=1e-7)):
            special_flags.add("Degeneracy Detected (One or more basic variables equal 0 in X_B)")
            
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
        if b_idx < num_vars:
            sol[b_idx] = rhs[pos]
    return status, sol, float(c @ sol), iterations, list(special_flags)

# ==========================================
# VIEW 1: LANDING SCREEN
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
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            obj_type = st.selectbox("Objective Type", ["Maximize", "Minimize"])
        with cfg2:
            num_vars = st.selectbox("Number of Decision Variables", [2, 3], index=0)
        with cfg3:
            num_cons = st.number_input("Number of Constraints", min_value=1, max_value=10, value=2, step=1)

        st.subheader("1. Objective Function")
        obj_cols = st.columns(num_vars)
        c_vals, default_c = [], [3.0, 5.0, 4.0]
        for j in range(num_vars):
            with obj_cols[j]:
                c_vals.append(st.number_input(f"Coeff of x{j+1}", value=default_c[j], key=f"c_{j}"))

        st.subheader("2. Constraints")
        A_rows, signs, b_vals = [], [], []
        default_A = [[1.0, 2.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 1.0]]
        default_b = [8.0, 5.0, 10.0]

        for i in range(int(num_cons)):
            row_cols = st.columns(num_vars + 2)
            row_a = []
            for j in range(num_vars):
                with row_cols[j]:
                    row_a.append(st.number_input(f"a[{i+1},{j+1}]", value=default_A[i][j] if i < 3 else 1.0, key=f"a_{i}_{j}"))
            with row_cols[num_vars]:
                signs.append(st.selectbox("Sign", ["<=", ">=", "="], key=f"sign_{i}"))
            with row_cols[num_vars + 1]:
                b_vals.append(st.number_input(f"RHS b{i+1}", value=default_b[i] if i < 3 else 6.0, key=f"b_{i}"))
            A_rows.append(row_a)

        status, sol, final_z, iterations, flags = solve_simplex_big_m(
            obj_type, np.array(c_vals), np.array(A_rows), signs, np.array(b_vals)
        )

        st.divider()
        st.subheader("3. Solution Status & Special Case Flags")
        if flags:
            for f in flags:
                if "Infeasible" in f or "Unbounded" in f:
                    st.error(f"🚩 **Special Case Flagged:** {f}")
                else:
                    st.warning(f"⚠️ **Special Case Flagged:** {f}")
        else:
            st.success("✅ **Unique Optimal Solution** (No degeneracy, infeasibility, unboundedness, or alternative optima detected).")

        if status == "Optimal":
            sol_str = ", ".join([f"**x{j+1} = {sol[j]:.4f}**" for j in range(num_vars)])
            st.info(f"**Optimal Objective Z = {final_z:.4f}** | {sol_str}")

        tab_col, viz_col = st.columns([1.3, 1], gap="large")
        with tab_col:
            st.subheader("4. Step-by-Step Simplex Tableau Construction")
            for iter_num, df, note in iterations:
                st.markdown(f"**Iteration {iter_num}** — {note}")
                st.dataframe(df, use_container_width=True)

        with viz_col:
            st.subheader("5. Graphical Visualization")
            if num_vars == 2:
                max_bound = max(max(b_vals) * 1.3, 10.0)
                x1_vals = np.linspace(0, max_bound, 250)
                x2_vals = np.linspace(0, max_bound, 250)
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
                for i in range(int(num_cons)):
                    a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
                    if abs(a2) > 1e-6:
                        fig.add_trace(go.Scatter(x=x1_vals, y=(b_i - a1*x1_vals)/a2, mode='lines', name=f'C{i+1}', line=dict(color=colors[i%len(colors)], width=2.5)))
                fig.add_trace(go.Contour(x=x1_vals, y=x2_vals, z=Z_space, colorscale='Plasma', showscale=True, opacity=0.85))
                if status == "Optimal":
                    fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode='markers+text', text=[f"Opt ({sol[0]:.2f}, {sol[1]:.2f})"], textposition="top right", marker=dict(size=13, color='#10b981', line=dict(width=2, color='white')), name="Optimal Point"))
                fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis_range=[0, max_bound], yaxis_range=[0, max_bound], template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)')
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 2. DUAL SIMPLEX METHOD
    # ------------------------------------------
    elif problem == "Dual Simplex Method":
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("1. Infeasible RHS Values")
            rhs1 = st.number_input("Constraint 1 RHS (negative)", value=-4.0)
            rhs2 = st.number_input("Constraint 2 RHS (negative)", value=-6.0)
            step = st.button("Run Pivot Step")
        with col2:
            st.subheader("2. Dual Simplex Tableau")
            tableau = np.array([[-3.0, -4.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 1.0, 0.0, rhs1], [-2.0, -1.0, 0.0, 1.0, rhs2]])
            st.dataframe(tableau, use_container_width=True)
            if step:
                st.success(f"Leaving Variable selected: **Row {'1' if rhs1 < rhs2 else '2'}** (most negative RHS).")

    # ------------------------------------------
    # 3. BRANCH AND BOUND (Interactive Variable Selection for Deletion Strip)
    # ------------------------------------------
    elif problem == "Branch and Bound":
        st.subheader("1. Continuous LP Relaxation Parameters")
        p1, p2, p3 = st.columns(3)
        with p1:
            c1 = st.number_input("Objective c₁ (x₁)", value=5.0)
            c2 = st.number_input("Objective c₂ (x₂)", value=4.0)
        with p2:
            x1_relax = st.number_input("Relaxed Optimal x₁*", value=2.4, step=0.1)
            x2_relax = st.number_input("Relaxed Optimal x₂*", value=1.8, step=0.1)
        with p3:
            # Interactive user choice on which variable to branch on when both are fractional
            branch_var = st.selectbox("Choose Variable to Branch On", ["x₁", "x₂"], help="Select which variable's fractional strip to delete.")
            
        val = x1_relax if branch_var == "x₁" else x2_relax
        floor_v = int(np.floor(val))
        ceil_v = int(np.ceil(val))
        z_relax = c1 * x1_relax + c2 * x2_relax

        st.divider()
        left_col, right_col = st.columns([1.2, 1], gap="large")
        
        with left_col:
            st.subheader(f"2. Strip Deletion Visualization ({branch_var})")
            st.caption(f"Deleting the fractional strip **{floor_v} < {branch_var} < {ceil_v}** as chosen by user.")
            
            x_grid = np.linspace(0, 6, 200)
            y_grid = np.linspace(0, 6, 200)
            X, Y = np.meshgrid(x_grid, y_grid)
            base_feas = (X >= 0) & (Y >= 0) & (X + Y <= x1_relax + x2_relax + 1.0)
            
            fig = go.Figure()
            sub1 = base_feas & ((X <= floor_v) if branch_var == "x₁" else (Y <= floor_v))
            sub2 = base_feas & ((X >= ceil_v) if branch_var == "x₁" else (Y >= ceil_v))
            
            fig.add_trace(go.Contour(x=x_grid, y=y_grid, z=sub1.astype(float), showscale=False, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(16, 185, 129, 0.35)']]))
            fig.add_trace(go.Contour(x=x_grid, y=y_grid, z=sub2.astype(float), showscale=False, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(56, 189, 248, 0.35)']]))

            if branch_var == "x₁":
                fig.add_vrect(x0=floor_v, x1=ceil_v, fillcolor="rgba(244, 63, 94, 0.28)", line_width=2, line_dash="dash", line_color="#f43f5e", annotation_text=f"Deleted Strip ({floor_v} < x₁ < {ceil_v})", annotation_position="top")
            else:
                fig.add_hrect(y0=floor_v, y1=ceil_v, fillcolor="rgba(244, 63, 94, 0.28)", line_width=2, line_dash="dash", line_color="#f43f5e", annotation_text=f"Deleted Strip ({floor_v} < x₂ < {ceil_v})", annotation_position="right")

            ix, iy = np.meshgrid(np.arange(0, 7), np.arange(0, 7))
            fig.add_trace(go.Scatter(x=ix.flatten(), y=iy.flatten(), mode='markers', marker=dict(size=6, color='rgba(255,255,255,0.45)'), name="Integer Lattice"))
            fig.add_trace(go.Scatter(x=[x1_relax], y=[x2_relax], mode='markers+text', text=[f"LP Opt ({x1_relax}, {x2_relax})"], textposition="top right", marker=dict(size=12, color='#fbbf24', line=dict(width=2, color='white')), name="Relaxed Optimum"))
            fig.update_layout(xaxis_title="x₁", yaxis_title="x₂", xaxis_range=[0, 5.5], yaxis_range=[0, 5.5], template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.6)')
            st.plotly_chart(fig, use_container_width=True)

        with right_col:
            st.subheader("3. New Sub-LPP Construction")
            st.markdown(f"""
            **Root Problem ($P_0$):**
            * $\\max Z = {c1}x_1 + {c2}x_2$ | $Z^* = {z_relax:.2f}$ at $({x1_relax}, {x2_relax})$
            """)
            st.success(f"""
            **Sub-Problem 1 ($P_1$):**
            * Subject to: Original Constraints **AND $\\mathbf{{{branch_var} \\le {floor_v}}}$**
            """)
            st.info(f"""
            **Sub-Problem 2 ($P_2$):**
            * Subject to: Original Constraints **AND $\\mathbf{{{branch_var} \\ge {ceil_v}}}$**
            """)

            st.subheader("4. Decision Tree Construction")
            dot = graphviz.Digraph()
            dot.attr(bgcolor='transparent', rankdir='TB')
            dot.node_attr.update(style='filled', fontcolor='white', shape='box', rounded='true')
            dot.node('P0', f'P0 (Root LP)\nZ = {z_relax:.2f}\n({x1_relax}, {x2_relax})', fillcolor='#6366f1')
