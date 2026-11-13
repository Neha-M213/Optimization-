import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fractions import Fraction
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
    label, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
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
        min-height: 60px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    div.stButton > button:hover {
        border-color: #818cf8;
        background: rgba(51, 65, 85, 0.9);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
        color: #38bdf8;
    }
    .cut-box {
        background: rgba(30, 41, 59, 0.85);
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "selected_problem" not in st.session_state:
    st.session_state.selected_problem = None
if "gomory_history" not in st.session_state:
    st.session_state.gomory_history = []
if "gomory_cuts_geom" not in st.session_state:
    st.session_state.gomory_cuts_geom = []
if "gomory_initialized" not in st.session_state:
    st.session_state.gomory_initialized = False

def select_problem(name):
    st.session_state.selected_problem = name

def go_home():
    st.session_state.selected_problem = None
    st.session_state.gomory_initialized = False

# ==========================================
# EXACT FRACTION HELPERS
# ==========================================
def to_frac(val):
    if isinstance(val, Fraction):
        return val
    return Fraction(str(val)).limit_denominator(10000)

def fmt_frac(val):
    f = to_frac(val)
    if abs(f) >= Fraction(10**5):
        return "-M" if f < 0 else "M"
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

def floor_frac(f):
    """Exact mathematical floor: e.g. floor(-1/3) = -1, floor(4/3) = 1"""
    f = to_frac(f)
    return Fraction(f.numerator // f.denominator, 1)

def frac_part(f):
    """Exact non-negative fractional part: f - floor(f), always in [0, 1)"""
    f = to_frac(f)
    return f - floor_frac(f)

# ==========================================
# EXACT SIMPLEX & DUAL SIMPLEX ENGINE
# ==========================================
def build_exact_tableau_df(state, dual_ratios=None):
    tableau = state["tableau"]
    rhs = state["rhs"]
    basis = state["basis"]
    var_names = state["var_names"]
    cost_row = state["cost_row"]
    obj_type = state["obj_type"]
    num_cons = len(basis)
    num_cols = len(var_names)

    cb = [cost_row[b] for b in basis]
    zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(num_cols)]
    cj_zj = [cost_row[j] - zj[j] for j in range(num_cols)]
    z_raw = sum(cb[i] * rhs[i] for i in range(num_cons))
    z_disp = z_raw if obj_type == "Maximize" else -z_raw

    cj_row = ["", "C_j ->", ""] + [fmt_frac(c) for c in cost_row] + [""]
    rows = [cj_row]

    for i in range(num_cons):
        r_str = dual_ratios[i] if dual_ratios is not None else "—"
        row_vals = [
            fmt_frac(cb[i]),
            var_names[basis[i]],
            fmt_frac(rhs[i])
        ] + [fmt_frac(tableau[i][j]) for j in range(num_cols)] + [r_str]
        rows.append(row_vals)

    zj_row = ["", "Z_j", fmt_frac(z_disp)] + [fmt_frac(v) for v in zj] + [""]
    cj_zj_row = ["", "C_j - Z_j", ""] + [fmt_frac(v) for v in cj_zj] + [""]
    rows.append(zj_row)
    rows.append(cj_zj_row)

    cols = ["C_B", "X_B", "b"] + var_names + ["Ratio"]
    return pd.DataFrame(rows, columns=cols)

def solve_lp_relaxation_exact(obj_type, c, A, signs, b):
    """Solves LP relaxation silently to return the optimal initial tableau for Gomory Cut."""
    M = Fraction(10**6, 1)
    num_vars = len(c)
    num_cons = len(b)

    A_f = [[to_frac(A[i][j]) for j in range(num_vars)] for i in range(num_cons)]
    b_f = [to_frac(b[i]) for i in range(num_cons)]
    signs_f = list(signs)

    for i in range(num_cons):
        if b_f[i] < 0:
            b_f[i] = -b_f[i]
            A_f[i] = [-x for x in A_f[i]]
            if signs_f[i] == "<=": signs_f[i] = ">="
            elif signs_f[i] == ">=": signs_f[i] = "<="

    c_eff = [to_frac(x) if obj_type == "Maximize" else -to_frac(x) for x in c]
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)

    extra_cols = []
    basis = []
    artificial_indices = []
    # Match notebook naming: decision vars x1..xn, slack/surplus x_{n+1}..
    next_x_idx = num_vars + 1

    for i, s in enumerate(signs_f):
        if s == "<=":
            col = [Fraction(0)] * num_cons
            col[i] = Fraction(1)
            extra_cols.append(col)
            var_names.append(f"x{next_x_idx}")
            cost_row.append(Fraction(0))
            basis.append(len(var_names) - 1)
            next_x_idx += 1
        elif s == ">=":
            col_s = [Fraction(0)] * num_cons
            col_s[i] = Fraction(-1)
            extra_cols.append(col_s)
            var_names.append(f"x{next_x_idx}")
            cost_row.append(Fraction(0))
            next_x_idx += 1

            col_a = [Fraction(0)] * num_cons
            col_a[i] = Fraction(1)
            extra_cols.append(col_a)
            var_names.append(f"a{i+1}")
            cost_row.append(-M)
            basis.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)
        elif s == "=":
            col_a = [Fraction(0)] * num_cons
            col_a[i] = Fraction(1)
            extra_cols.append(col_a)
            var_names.append(f"a{i+1}")
            cost_row.append(-M)
            basis.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)

    num_extra = len(extra_cols)
    tableau = [
        A_f[i] + [extra_cols[k][i] for k in range(num_extra)]
        for i in range(num_cons)
    ]
    rhs = list(b_f)
    num_cols = len(var_names)

    # Primal Simplex to reach LP Optimum
    for _ in range(100):
        cb = [cost_row[b_idx] for b_idx in basis]
        zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(num_cols)]
        cj_zj = [cost_row[j] - zj[j] for j in range(num_cols)]

        max_val = max(cj_zj)
        if max_val <= 0:
            break
        enter_col = cj_zj.index(max_val)

        best_ratio = None
        leave_row = -1
        for i in range(num_cons):
            if tableau[i][enter_col] > 0:
                ratio = rhs[i] / tableau[i][enter_col]
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    leave_row = i

        if leave_row == -1:
            return None, "LP Relaxation is Unbounded!"

        # Pivot
        piv = tableau[leave_row][enter_col]
        tableau[leave_row] = [x / piv for x in tableau[leave_row]]
        rhs[leave_row] /= piv

        for i in range(num_cons):
            if i != leave_row:
                factor = tableau[i][enter_col]
                tableau[i] = [tableau[i][j] - factor * tableau[leave_row][j] for j in range(num_cols)]
                rhs[i] -= factor * rhs[leave_row]

        basis[leave_row] = enter_col

    # Check artificials
    for i, b_idx in enumerate(basis):
        if b_idx in artificial_indices and rhs[i] > 0:
            return None, "LP Relaxation is Infeasible!"

    # Remove artificial columns cleanly if all left basis
    keep_cols = [j for j in range(num_cols) if j not in artificial_indices]
    col_map = {old_j: new_j for new_j, old_j in enumerate(keep_cols)}

    clean_var_names = [var_names[j] for j in keep_cols]
    clean_cost_row = [cost_row[j] for j in keep_cols]
    clean_tableau = [[tableau[i][j] for j in keep_cols] for i in range(num_cons)]
    clean_basis = [col_map[b_idx] for b_idx in basis]

    return {
        "tableau": clean_tableau,
        "rhs": rhs,
        "basis": clean_basis,
        "var_names": clean_var_names,
        "cost_row": clean_cost_row,
        "obj_type": obj_type,
        "num_decision_vars": num_vars,
        "cut_count": 0
    }, None

def apply_gomory_cut_and_dual_simplex(state, chosen_row_idx, orig_A, orig_b, orig_signs):
    """
    1. Generates Gomory fractional cut from chosen_row_idx.
    2. Expresses the cut in terms of original decision variables (x1, x2, ...) for plotting.
    3. Appends cut row & slack s_k to tableau.
    4. Runs Dual Simplex steps until feasibility (b >= 0) is restored.
    """
    tableau = [list(r) for r in state["tableau"]]
    rhs = list(state["rhs"])
    basis = list(state["basis"])
    var_names = list(state["var_names"])
    cost_row = list(state["cost_row"])
    num_cons = len(basis)
    num_cols = len(var_names)
    cut_num = state["cut_count"] + 1
    slack_name = f"s{cut_num}"

    # --- Step 1: Extract chosen row & compute fractional parts ---
    row_coeffs = tableau[chosen_row_idx]
    b_val = rhs[chosen_row_idx]
    basic_var_name = var_names[basis[chosen_row_idx]]

    f_b = frac_part(b_val)
    f_coeffs = [frac_part(a) for a in row_coeffs]

    # Build LaTeX derivation strings matching handwritten notes
    orig_terms = [f"{fmt_frac(b_val)} = {basic_var_name}"]
    split_terms = [f"({fmt_frac(floor_frac(b_val))} + {fmt_frac(f_b)}) = {basic_var_name}"]
    cut_terms = []

    for j in range(num_cols):
        if j == basis[chosen_row_idx]:
            continue
        a_ij = row_coeffs[j]
        if a_ij != 0:
            sign_str = "+" if a_ij > 0 else "-"
            orig_terms.append(f"{sign_str} {fmt_frac(abs(a_ij))}{var_names[j]}")
            fl = floor_frac(a_ij)
            fr = f_coeffs[j]
            split_terms.append(f"+ ({fmt_frac(fl)} + {fmt_frac(fr)}){var_names[j]}")
        if f_coeffs[j] > 0:
            cut_terms.append(f"-\\frac{{{f_coeffs[j].numerator}}}{{{f_coeffs[j].denominator}}}{var_names[j]}" if f_coeffs[j].denominator != 1 else f"-{f_coeffs[j].numerator}{var_names[j]}")

    cut_eq_latex = " ".join(cut_terms) + f" + {slack_name} = -{fmt_frac(f_b)}"

    # --- Step 2: Convert cut to original decision variables x1..xn for 2D/3D Geometry ---
    # Any slack variable x_{n+1} = b_orig - sum(A_orig * x), surplus x_{n+1} = sum(A_orig * x) - b_orig
    # Previous Gomory slacks s_m satisfy: s_m = -f_b_prev + sum(f_prev * vars)
    n_dec = state["num_decision_vars"]
    # Express every variable in `var_names` as: var_j = const_j + sum_{d=0}^{n_dec-1} coeff_{j,d} * x_{d+1}
    var_expr_const = [Fraction(0)] * num_cols
    var_expr_coeffs = [[Fraction(0)] * n_dec for _ in range(num_cols)]

    for d in range(n_dec):
        var_expr_coeffs[d][d] = Fraction(1)

    # Map original constraint slacks/surpluses
    col_ptr = n_dec
    for i, s in enumerate(orig_signs):
        b_i = to_frac(orig_b[i])
        a_i = [to_frac(x) for x in orig_A[i]]
        if s == "<=":
            # x_slack = b_i - sum(a_i * x)
            var_expr_const[col_ptr] = b_i
            var_expr_coeffs[col_ptr] = [-x for x in a_i]
            col_ptr += 1
        elif s == ">=":
            # x_surplus = sum(a_i * x) - b_i
            var_expr_const[col_ptr] = -b_i
            var_expr_coeffs[col_ptr] = list(a_i)
            col_ptr += 1

    # Map any previously added Gomory slacks stored in state
    prev_slack_exprs = state.get("slack_exprs", {})
    for idx_col in range(col_ptr, num_cols):
        vname = var_names[idx_col]
        if vname in prev_slack_exprs:
            var_expr_const[idx_col], var_expr_coeffs[idx_col] = prev_slack_exprs[vname]

    # Now compute the new slack s_k = -f_b + sum_{j not basic} f_coeffs[j] * var_j >= 0
    s_const = -f_b
    s_coeffs = [Fraction(0)] * n_dec
    for j in range(num_cols):
        if j != basis[chosen_row_idx] and f_coeffs[j] != 0:
            s_const += f_coeffs[j] * var_expr_const[j]
            for d in range(n_dec):
                s_coeffs[d] += f_coeffs[j] * var_expr_coeffs[j][d]

    # Inequality in original variables: sum(s_coeffs[d] * x_d) + s_const >= 0
    # Or equivalently: sum(-s_coeffs[d] * x_d) <= s_const
    geom_cut = {
        "name": f"Cut {cut_num} ({slack_name})",
        "a": [float(-x) for x in s_coeffs],
        "b": float(s_const),
        "latex_orig": " + ".join([f"({fmt_frac(-s_coeffs[d])})x_{d+1}" for d in range(n_dec)]) + f" \\le {fmt_frac(s_const)}"
    }

    new_slack_exprs = dict(prev_slack_exprs)
    new_slack_exprs[slack_name] = (s_const, s_coeffs)

    # --- Step 3: Append Cut Row to Tableau ---
    for i in range(num_cons):
        tableau[i].append(Fraction(0))

    new_row = [-f_coeffs[j] if j != basis[chosen_row_idx] else Fraction(0) for j in range(num_cols)] + [Fraction(1)]
    tableau.append(new_row)
    rhs.append(-f_b)
    var_names.append(slack_name)
    cost_row.append(Fraction(0))
    basis.append(len(var_names) - 1)

    num_cons += 1
    num_cols += 1

    # --- Step 4: Dual Simplex Iterations ---
    dual_steps = []
    for _ in range(30):
        # Find most negative RHS
        min_b = min(rhs)
        if min_b >= 0:
            # Feasible! Record final tableau of this cut
            final_state = {
                "tableau": [list(r) for r in tableau],
                "rhs": list(rhs),
                "basis": list(basis),
                "var_names": list(var_names),
                "cost_row": list(cost_row),
                "obj_type": state["obj_type"],
                "num_decision_vars": n_dec,
                "cut_count": cut_num,
                "slack_exprs": new_slack_exprs
            }
            dual_steps.append({
                "df": build_exact_tableau_df(final_state),
                "note": "All $b_i \\ge 0$: Dual Simplex optimality & feasibility restored."
            })
            return final_state, {
                "cut_num": cut_num,
                "basic_var": basic_var_name,
                "orig_row_str": " ".join(orig_terms),
                "split_row_str": " ".join(split_terms),
                "cut_eq_latex": cut_eq_latex,
                "geom_cut": geom_cut,
                "dual_steps": dual_steps
            }, None

        leave_row = rhs.index(min_b)
        cb = [cost_row[b_idx] for b_idx in basis]
        zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(num_cols)]
        cj_zj = [cost_row[j] - zj[j] for j in range(num_cols)]

        # Compute Dual Ratio = |(C_j - Z_j) / a_{leave_row, j}| for a_{leave_row, j} < 0
        ratios_disp = ["—"] * num_cons
        best_ratio = None
        enter_col = -1
        col_ratio_strs = []

        for j in range(num_cols):
            a_rj = tableau[leave_row][j]
            if a_rj < 0:
                ratio = abs(cj_zj[j] / a_rj)
                col_ratio_strs.append(f"{var_names[j]}: {fmt_frac(ratio)}")
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    enter_col = j

        if enter_col == -1:
            return None, None, "Integer Problem is Infeasible (no negative entry in leaving row during Dual Simplex)."

        snap_state = {
            "tableau": [list(r) for r in tableau],
            "rhs": list(rhs),
            "basis": list(basis),
            "var_names": list(var_names),
            "cost_row": list(cost_row),
            "obj_type": state["obj_type"]
        }
        # Show Dual Ratios on the leaving row or summary note
        ratios_disp[leave_row] = f"Min -> {var_names[enter_col]} ({fmt_frac(best_ratio)})"
        dual_steps.append({
            "df": build_exact_tableau_df(snap_state, dual_ratios=ratios_disp),
            "note": f"**Leaving Variable:** `{var_names[basis[leave_row]]}` ($b = {fmt_frac(rhs[leave_row])}$) | **Dual Ratios:** {', '.join(col_ratio_strs)} | **Entering Variable:** `{var_names[enter_col]}`"
        })

        # Pivot
        piv = tableau[leave_row][enter_col]
        tableau[leave_row] = [x / piv for x in tableau[leave_row]]
        rhs[leave_row] /= piv

        for i in range(num_cons):
            if i != leave_row:
                factor = tableau[i][enter_col]
                tableau[i] = [tableau[i][j] - factor * tableau[leave_row][j] for j in range(num_cols)]
                rhs[i] -= factor * rhs[leave_row]

        basis[leave_row] = enter_col

    return None, None, "Maximum Dual Simplex iterations reached."

# ==========================================
# 2D & 3D GEOMETRIC VISUALIZER
# ==========================================
def plot_gomory_2d(A, b, signs, cuts_geom, current_pt, is_integer_opt):
    fig = go.Figure()

    # Collect all inequalities in form a1*x1 + a2*x2 <= rhs
    ineqs = []
    for i in range(len(b)):
        if signs[i] == "<=":
            ineqs.append((float(A[i][0]), float(A[i][1]), float(b[i]), f"C{i+1}"))
        elif signs[i] == ">=":
            ineqs.append((-float(A[i][0]), -float(A[i][1]), -float(b[i]), f"C{i+1}"))
        elif signs[i] == "=":
            ineqs.append((float(A[i][0]), float(A[i][1]), float(b[i]), f"C{i+1}"))
            ineqs.append((-float(A[i][0]), -float(A[i][1]), -float(b[i]), f"C{i+1}"))

    # Non-negativity
    ineqs.append((-1.0, 0.0, 0.0, "x1>=0"))
    ineqs.append((0.0, -1.0, 0.0, "x2>=0"))

    # Add previous cuts vs latest cut
    all_ineqs = list(ineqs)
    for c in cuts_geom:
        all_ineqs.append((c["a"][0], c["a"][1], c["b"], c["name"]))

    def get_polygon_vertices(ineq_list):
        pts = []
        for (a1, a2, rhs1, _), (b1, b2, rhs2, _) in itertools.combinations(ineq_list, 2):
            det = a1 * b2 - a2 * b1
            if abs(det) > 1e-9:
                x = (rhs1 * b2 - rhs2 * a2) / det
                y = (a1 * rhs2 - b1 * rhs1) / det
                if all(ia * x + ib * y <= irhs + 1e-6 for ia, ib, irhs, _ in ineq_list):
                    pts.append((x, y))
        if not pts:
            return []
        # Deduplicate & sort counterclockwise
        unique_pts = []
        for p in pts:
            if not any(np.hypot(p[0]-u[0], p[1]-u[1]) < 1e-5 for u in unique_pts):
                unique_pts.append(p)
        cx = sum(p[0] for p in unique_pts) / len(unique_pts)
        cy = sum(p[1] for p in unique_pts) / len(unique_pts)
        unique_pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        return unique_pts

    orig_verts = get_polygon_vertices(ineqs)
    curr_verts = get_polygon_vertices(all_ineqs)

    max_x = max([p[0] for p in orig_verts] + [current_pt[0], 4.0]) + 1.0
    max_y = max([p[1] for p in orig_verts] + [current_pt[1], 4.0]) + 1.0

    # 1. Original LP Relaxation Polygon (Faded)
    if len(orig_verts) >= 3:
        ox = [p[0] for p in orig_verts] + [orig_verts[0][0]]
        oy = [p[1] for p in orig_verts] + [orig_verts[0][1]]
        fig.add_trace(go.Scatter(
            x=ox, y=oy, fill="toself",
            fillcolor="rgba(148, 163, 184, 0.12)",
            line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5),
            name="Initial LP Relaxation"
        ))

    # 2. Current Trimmed Feasible Region
    if len(curr_verts) >= 3:
        cx_p = [p[0] for p in curr_verts] + [curr_verts[0][0]]
        cy_p = [p[1] for p in curr_verts] + [curr_verts[0][1]]
        fig.add_trace(go.Scatter(
            x=cx_p, y=cy_p, fill="toself",
            fillcolor="rgba(56, 189, 248, 0.25)",
            line=dict(color="#38bdf8", width=2.5),
            name="Current Trimmed Feasible Region"
        ))

    # 3. Integer Lattice Points inside Original LP
    gx = np.arange(0, int(np.ceil(max_x)) + 1)
    gy = np.arange(0, int(np.ceil(max_y)) + 1)
    feas_ix, feas_iy, infeas_ix, infeas_iy = [], [], [], []
    for ix in gx:
        for iy in gy:
            if all(ia * ix + ib * iy <= irhs + 1e-6 for ia, ib, irhs, _ in ineqs):
                feas_ix.append(ix)
                feas_iy.append(iy)
            else:
                infeas_ix.append(ix)
                infeas_iy.append(iy)

    fig.add_trace(go.Scatter(
        x=infeas_ix, y=infeas_iy, mode="markers",
        marker=dict(color="rgba(148, 163, 184, 0.25)", size=6),
        name="Infeasible Grid Points"
    ))
    fig.add_trace(go.Scatter(
        x=feas_ix, y=feas_iy, mode="markers",
        marker=dict(color="#34d399", size=10, line=dict(color="#064e3b", width=1.5)),
        name="Feasible Integer Points"
    ))

    # 4. Plot Cutting Lines
    x_line = np.linspace(-0.5, max_x, 200)
    for idx, c in enumerate(cuts_geom):
        a1, a2, rhs_c = c["a"][0], c["a"][1], c["b"]
        is_latest = (idx == len(cuts_geom) - 1)
        color = "#f43f5e" if is_latest else "#fb923c"
        dash = "dash" if is_latest else "dot"
        if abs(a2) > 1e-9:
            y_line = (rhs_c - a1 * x_line) / a2
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=color, width=2.5 if is_latest else 1.8, dash=dash),
                name=c["name"]
            ))
        elif abs(a1) > 1e-9:
            x_val = rhs_c / a1
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[-0.5, max_y], mode="lines",
                line=dict(color=color, width=2.5 if is_latest else 1.8, dash=dash),
                name=c["name"]
            ))

    # 5. Current Basic Optimal Point
    pt_color = "#10b981" if is_integer_opt else "#f43f5e"
    pt_symbol = "diamond" if is_integer_opt else "star"
    pt_label = f"Integer Optimum ({current_pt[0]:g}, {current_pt[1]:g})" if is_integer_opt else f"Current Vertex ({current_pt[0]:.2f}, {current_pt[1]:.2f})"
    fig.add_trace(go.Scatter(
        x=[current_pt[0]], y=[current_pt[1]], mode="markers+text",
        text=[pt_label], textposition="top right",
        textfont=dict(color=pt_color, size=13),
        marker=dict(color=pt_color, size=15, symbol=pt_symbol, line=dict(color="white", width=1.5)),
        name=pt_label
    ))

    fig.update_layout(
        title="2D Geometric Visualization: Polytope Shaving & Integer Lattice",
        xaxis=dict(title="x₁", range=[-0.3, max_x], gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.4)"),
        yaxis=dict(title="x₂", range=[-0.3, max_y], gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.4)"),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        font=dict(color="#f8fafc"),
        height=520,
        legend=dict(bgcolor="rgba(30, 41, 59, 0.8)")
    )
    return fig

def plot_gomory_3d(A, b, signs, cuts_geom, current_pt, is_integer_opt):
    fig = go.Figure()

    ineqs = []
    for i in range(len(b)):
        if signs[i] == "<=":
            ineqs.append((float(A[i][0]), float(A[i][1]), float(A[i][2]), float(b[i])))
        elif signs[i] == ">=":
            ineqs.append((-float(A[i][0]), -float(A[i][1]), -float(A[i][2]), -float(b[i])))
        elif signs[i] == "=":
            ineqs.append((float(A[i][0]), float(A[i][1]), float(A[i][2]), float(b[i])))
            ineqs.append((-float(A[i][0]), -float(A[i][1]), -float(A[i][2]), -float(b[i])))

    ineqs.append((-1.0, 0.0, 0.0, 0.0))
    ineqs.append((0.0, -1.0, 0.0, 0.0))
    ineqs.append((0.0, 0.0, -1.0, 0.0))

    all_ineqs = list(ineqs)
    for c in cuts_geom:
        all_ineqs.append((c["a"][0], c["a"][1], c["a"][2], c["b"]))

    def get_3d_vertices(ineq_list):
        pts = []
        for c1, c2, c3 in itertools.combinations(ineq_list, 3):
            M = np.array([[c1[0], c1[1], c1[2]], [c2[0], c2[1], c2[2]], [c3[0], c3[1], c3[2]]])
            rhs_v = np.array([c1[3], c2[3], c3[3]])
            if abs(np.linalg.det(M)) > 1e-8:
                sol = np.linalg.solve(M, rhs_v)
                if all(ia*sol[0] + ib*sol[1] + ic*sol[2] <= irhs + 1e-5 for ia, ib, ic, irhs in ineq_list):
                    pts.append(sol)
        unique_pts = []
        for p in pts:
            if not any(np.linalg.norm(p - u) < 1e-4 for u in unique_pts):
                unique_pts.append(p)
        return np.array(unique_pts)

    orig_verts = get_3d_vertices(ineqs)
    curr_verts = get_3d_vertices(all_ineqs)

    max_coord = 4.0
    if len(orig_verts) > 0:
        max_coord = max(max_coord, float(np.max(orig_verts)) + 1.0)
    max_coord = max(max_coord, float(max(current_pt)) + 1.0)

    # 1. Current Trimmed 3D Polyhedron Convex Hull
    if len(curr_verts) >= 4:
        fig.add_trace(go.Mesh3d(
            x=curr_verts[:, 0], y=curr_verts[:, 1], z=curr_verts[:, 2],
            alphahull=0,
            opacity=0.28,
            color="#38bdf8",
            name="Trimmed Feasible Polyhedron"
        ))
        fig.add_trace(go.Scatter3d(
            x=curr_verts[:, 0], y=curr_verts[:, 1], z=curr_verts[:, 2],
            mode="markers",
            marker=dict(size=4, color="#38bdf8"),
            name="Polyhedron Vertices"
        ))

    # 2. Feasible 3D Integer Lattice Points
    lim = int(np.ceil(max_coord))
    ix_list, iy_list, iz_list = [], [], []
    for ix in range(lim + 1):
        for iy in range(lim + 1):
            for iz in range(lim + 1):
                if all(ia*ix + ib*iy + ic*iz <= irhs + 1e-5 for ia, ib, ic, irhs in ineqs):
                    ix_list.append(ix)
                    iy_list.append(iy)
                    iz_list.append(iz)

    fig.add_trace(go.Scatter3d(
        x=ix_list, y=iy_list, z=iz_list,
        mode="markers",
        marker=dict(size=6, color="#34d399", opacity=0.9),
        name="Feasible Integer Lattice"
    ))

    # 3. Render Cutting Planes in 3D
    grid = np.linspace(0, max_coord, 15)
    for idx, c in enumerate(cuts_geom):
        a1, a2, a3, rhs_c = c["a"][0], c["a"][1], c["a"][2], c["b"]
        is_latest = (idx == len(cuts_geom) - 1)
        colorscale = [[0, "#f43f5e"], [1, "#f43f5e"]] if is_latest else [[0, "#fb923c"], [1, "#fb923c"]]

        if abs(a3) > 1e-6:
            X, Y = np.meshgrid(grid, grid)
            Z = (rhs_c - a1 * X - a2 * Y) / a3
            Z = np.where((Z >= -0.2) & (Z <= max_coord), Z, np.nan)
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.35 if is_latest else 0.18,
                colorscale=colorscale, name=c["name"]
            ))
        elif abs(a2) > 1e-6:
            X, Z = np.meshgrid(grid, grid)
            Y = (rhs_c - a1 * X - a3 * Z) / a2
            Y = np.where((Y >= -0.2) & (Y <= max_coord), Y, np.nan)
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.35 if is_latest else 0.18,
                colorscale=colorscale, name=c["name"]
            ))
        elif abs(a1) > 1e-6:
            Y, Z = np.meshgrid(grid, grid)
            X = (rhs_c - a2 * Y - a3 * Z) / a1
            X = np.where((X >= -0.2) & (X <= max_coord), X, np.nan)
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.35 if is_latest else 0.18,
                colorscale=colorscale, name=c["name"]
            ))

    # 4. Current Optimal Vertex
    pt_color = "#10b981" if is_integer_opt else "#f43f5e"
    pt_label = f"Integer Optimum ({current_pt[0]:g}, {current_pt[1]:g}, {current_pt[2]:g})" if is_integer_opt else f"Current Vertex ({current_pt[0]:.2f}, {current_pt[1]:.2f}, {current_pt[2]:.2f})"
    fig.add_trace(go.Scatter3d(
        x=[current_pt[0]], y=[current_pt[1]], z=[current_pt[2]],
        mode="markers+text",
        text=[pt_label],
        textfont=dict(color=pt_color, size=12),
        marker=dict(size=9, color=pt_color, symbol="diamond"),
        name=pt_label
    ))

    fig.update_layout(
        title="3D Geometric Visualization: Polyhedron Cutting Planes & Integer Lattice",
        scene=dict(
            xaxis=dict(title="x₁", range=[0, max_coord], backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(148,163,184,0.2)"),
            yaxis=dict(title="x₂", range=[0, max_coord], backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(148,163,184,0.2)"),
            zaxis=dict(title="x₃", range=[0, max_coord], backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(148,163,184,0.2)"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#f8fafc"),
        height=600,
        legend=dict(bgcolor="rgba(30, 41, 59, 0.8)")
    )
    return fig

# ==========================================
# GOMORY FRACTIONAL CUT MODULE UI
# ==========================================
def render_gomory_cut_studio():
    st.button("← Back to Studio Home", on_click=go_home)
    st.markdown("## ✂️ Gomory's Fractional Cut Method (Interactive Solver & 2D/3D Visualizer)")
    st.markdown("Configure any Integer Linear Programming Problem below. The solver jumps straight to the **Optimal LP Relaxation Tableau** and lets you choose which fractional row to slice at every step.")

    # --- FLEXIBLE PROBLEM INPUT ---
    with st.expander("📝 Problem Setup (Flexible Dimensions & Presets)", expanded=not st.session_state.gomory_initialized):
        col_p1, col_p2, col_p3 = st.columns([1.2, 1, 1])
        with col_p1:
            preset = st.selectbox(
                "Load Preset or Custom",
                [
                    "Notebook Example (2D: Max x₁ + x₂)",
                    "3D Integer Cube Slice (3D: Max 3x₁ + 4x₂ + 2x₃)",
                    "Custom Problem"
                ]
            )
        with col_p2:
            if preset == "Notebook Example (2D: Max x₁ + x₂)":
                default_vars, default_cons = 2, 2
            elif preset == "3D Integer Cube Slice (3D: Max 3x₁ + 4x₂ + 2x₃)":
                default_vars, default_cons = 3, 3
            else:
                default_vars, default_cons = 2, 2

            num_vars = st.number_input("Number of Decision Variables (n)", min_value=2, max_value=6, value=default_vars, step=1)
        with col_p3:
            num_cons = st.number_input("Number of Constraints (m)", min_value=1, max_value=8, value=default_cons, step=1)

        obj_type = st.radio("Objective Type", ["Maximize", "Minimize"], horizontal=True)

        st.markdown("**Objective Function Coefficients ($Z$):**")
        obj_cols = st.columns(num_vars)
        c_vals = []
        for j in range(num_vars):
            def_c = 1.0
            if preset == "Notebook Example (2D: Max x₁ + x₂)" and j < 2:
                def_c = [1.0, 1.0][j]
            elif preset == "3D Integer Cube Slice (3D: Max 3x₁ + 4x₂ + 2x₃)" and j < 3:
                def_c = [3.0, 4.0, 2.0][j]
            with obj_cols[j]:
                c_vals.append(st.number_input(f"c_{j+1} (x{j+1})", value=def_c, step=0.5, key=f"gom_c_{j}"))

        st.markdown("**Subject to Constraints ($A\\mathbf{x} \\le / = / \\ge \\mathbf{b}$):**")
        A_vals, signs_vals, b_vals = [], [], []
        for i in range(num_cons):
            row_cols = st.columns(num_vars + 2)
            row_a = []
            for j in range(num_vars):
                def_a = 1.0
                if preset == "Notebook Example (2D: Max x₁ + x₂)" and i < 2 and j < 2:
                    def_a = [[2.0, 1.0], [1.0, 2.0]][i][j]
                elif preset == "3D Integer Cube Slice (3D: Max 3x₁ + 4x₂ + 2x₃)" and i < 3 and j < 3:
                    def_a = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]][i][j]
                with row_cols[j]:
                    row_a.append(st.number_input(f"a_{i+1},{j+1}", value=def_a, step=0.5, key=f"gom_a_{i}_{j}", label_visibility="collapsed"))
            with row_cols[num_vars]:
                signs_vals.append(st.selectbox(f"Sign {i+1}", ["<=", ">=", "="], index=0, key=f"gom_s_{i}", label_visibility="collapsed"))
            with row_cols[num_vars + 1]:
                def_b = 4.0
                if preset == "Notebook Example (2D: Max x₁ + x₂)" and i < 2:
                    def_b = [4.0, 4.0][i]
                elif preset == "3D Integer Cube Slice (3D: Max 3x₁ + 4x₂ + 2x₃)" and i < 3:
                    def_b = [5.0, 5.0, 5.0][i]
                b_vals.append(st.number_input(f"b_{i+1}", value=def_b, step=0.5, key=f"gom_b_{i}", label_visibility="collapsed"))
            A_vals.append(row_a)

        if st.button("🚀 Jump to Optimal LP Relaxation Tableau", type="primary"):
            init_state, err = solve_lp_relaxation_exact(obj_type, c_vals, A_vals, signs_vals, b_vals)
            if err:
                st.error(err)
            else:
                st.session_state.gomory_history = [{
                    "state": init_state,
                    "cut_info": None
                }]
                st.session_state.gomory_cuts_geom = []
                st.session_state.gomory_problem = {
                    "obj_type": obj_type,
                    "c": c_vals,
                    "A": A_vals,
                    "signs": signs_vals,
                    "b": b_vals,
                    "num_vars": num_vars
                }
                st.session_state.gomory_initialized = True
                st.rerun()

    # --- INTERACTIVE SOLVING & VISUALIZATION WORKSPACE ---
    if st.session_state.gomory_initialized and st.session_state.gomory_history:
        prob = st.session_state.gomory_problem
        history = st.session_state.gomory_history
        current_entry = history[-1]
        curr_state = current_entry["state"]

        # Extract current decision variable values
        dec_vals = [Fraction(0)] * prob["num_vars"]
        for row_i, b_idx in enumerate(curr_state["basis"]):
            if b_idx < prob["num_vars"]:
                dec_vals[b_idx] = curr_state["rhs"][row_i]

        # Check integrality of ALL basic variables (or decision variables)
        fractional_rows = []
        for row_i, b_idx in enumerate(curr_state["basis"]):
            if frac_part(curr_state["rhs"][row_i]) != 0:
                fractional_rows.append((row_i, curr_state["var_names"][b_idx], curr_state["rhs"][row_i]))

        is_integer_opt = (len(fractional_rows) == 0)

        # Layout: Left Column = Tableau & Step-by-Step Cut; Right Column = 2D/3D Geometry
        col_left, col_right = st.columns([1.25, 1])

        with col_left:
            st.markdown("### 📋 Step 1: Optimal LP Relaxation Tableau")
            st.dataframe(build_exact_tableau_df(history[0]["state"]), use_container_width=True, hide_index=True)

            # Display all cuts generated so far
            for step_idx in range(1, len(history)):
                info = history[step_idx]["cut_info"]
                st.markdown(f"---")
                st.markdown(f"### ✂️ Gomory Fractional Cut #{info['cut_num']} (Generated from `{info['basic_var']}` row)")
                st.markdown(f"""
                <div class="cut-box">
                    <b>1. Source Row Equation:</b><br>
                    $${info['orig_row_str']}$$
                    <b>2. Separate into Integer + Non-negative Fractional Parts ($f_i \\in [0, 1)$):</b><br>
                    $${info['split_row_str']}$$
                    <b>3. Required Fractional Cut Constraint:</b><br>
                    $${info['cut_eq_latex']}$$
                    <b>4. Cut in Original Coordinates:</b> $${info['geom_cut']['latex_orig']}$$
                </div>
                """, unsafe_allow_html=True)

                for d_idx, d_step in enumerate(info["dual_steps"]):
                    st.markdown(f"**Dual Simplex Table {info['cut_num']}.{d_idx + 1}:** {d_step['note']}")
                    st.dataframe(d_step["df"], use_container_width=True, hide_index=True)

            # --- INTERACTIVE ROW SELECTOR OR FINAL OPTIMUM ---
            st.markdown("---")
            if is_integer_opt:
                z_opt = sum(to_frac(prob["c"][d]) * dec_vals[d] for d in range(prob["num_vars"]))
                sol_str = ", ".join([f"$x_{d+1} = {fmt_frac(dec_vals[d])}$" for d in range(prob["num_vars"])])
                st.success(f"🎉 **All-Integer Optimal Solution Reached!**\n\n{sol_str} | **Optimal $Z = {fmt_frac(z_opt)}$**")
                if len(history) > 1:
                    if st.button("↩️ Undo Last Cut"):
                        st.session_state.gomory_history.pop()
                        st.session_state.gomory_cuts_geom.pop()
                        st.rerun()
            else:
                st.markdown("### 🎯 Choose Row for Next Fractional Cut")
                row_options = {
                    f"Row {r_i + 1}: {v_name} = {fmt_frac(r_val)}  (fractional part = {fmt_frac(frac_part(r_val))})": r_i
                    for r_i, v_name, r_val in fractional_rows
                }
                # Default recommend largest fractional part
                default_choice = max(range(len(fractional_rows)), key=lambda idx: frac_part(fractional_rows[idx][2]))
                selected_label = st.selectbox(
                    "Select a basic variable row with a fractional RHS:",
                    list(row_options.keys()),
                    index=default_choice
                )
                chosen_row = row_options[selected_label]

                c_btn1, c_btn2 = st.columns([2, 1])
                with c_btn1:
                    if st.button("✂️ Generate Cut from Selected Row & Apply Dual Simplex", type="primary"):
                        next_state, cut_info, err = apply_gomory_cut_and_dual_simplex(
                            curr_state, chosen_row, prob["A"], prob["b"], prob["signs"]
                        )
                        if err:
                            st.error(err)
                        else:
                            st.session_state.gomory_history.append({
                                "state": next_state,
                                "cut_info": cut_info
                            })
                            st.session_state.gomory_cuts_geom.append(cut_info["geom_cut"])
                            st.rerun()
                with c_btn2:
                    if len(history) > 1:
                        if st.button("↩️ Undo Last Cut"):
                            st.session_state.gomory_history.pop()
                            st.session_state.gomory_cuts_geom.pop()
                            st.rerun()

        # --- RIGHT COLUMN: 2D / 3D GEOMETRY ---
        with col_right:
            st.markdown("### 🌐 Feasible Region & Cutting Plane Geometry")
            curr_pt_float = [float(x) for x in dec_vals]
            if prob["num_vars"] == 2:
                fig = plot_gomory_2d(
                    prob["A"], prob["b"], prob["signs"],
                    st.session_state.gomory_cuts_geom,
                    curr_pt_float, is_integer_opt
                )
                st.plotly_chart(fig, use_container_width=True)
            elif prob["num_vars"] == 3:
                fig = plot_gomory_3d(
                    prob["A"], prob["b"], prob["signs"],
                    st.session_state.gomory_cuts_geom,
                    curr_pt_float, is_integer_opt
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Geometric plotting is available for **2D** ($n=2$) and **3D** ($n=3$) problems. Your current problem has $n={prob['num_vars']}$ decision variables—all algebraic cuts and Dual Simplex tableaux on the left continue to work identically!")

            # Live Coordinate Status Card
            coord_lines = [f"- **$x_{d+1}$** = `{fmt_frac(dec_vals[d])}` ({float(dec_vals[d]):.4f})" for d in range(prob["num_vars"])]
            z_curr = sum(to_frac(prob["c"][d]) * dec_vals[d] for d in range(prob["num_vars"]))
            st.markdown("#### 📌 Current Vertex Coordinates")
            st.markdown("\n".join(coord_lines) + f"\n- **Objective $Z$** = `{fmt_frac(z_curr)}` ({float(z_curr):.4f})")

# ==========================================
# MAIN ROUTER / HOME SCREEN
# ==========================================
if st.session_state.selected_problem == "gomory_cut":
    render_gomory_cut_studio()
else:
    st.markdown('<div class="hero-title">Operations Research Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Interactive Optimization, Cutting Planes & Network Visualizers</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✂️ Gomory Fractional Cut\n(2D / 3D Visualizer & Interactive Rows)"):
            select_problem("gomory_cut")
            st.rerun()
    with c2:
        if st.button("📐 Simplex & Dual Simplex\n(Coming / Plug your existing module here)"):
            st.info("Connect your existing Simplex view to this button.")
    with c3:
        if st.button("🌳 Branch & Bound / TSP\n(Coming / Plug your existing module here)"):
            st.info("Connect your existing Branch & Bound / TSP view to this button.")
