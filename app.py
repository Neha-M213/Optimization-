import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import graphviz
from fractions import Fraction
import math

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
# HELPER: EXACT FRACTION CONVERTER & FORMATTER
# ==========================================
def to_frac(val):
    if isinstance(val, Fraction):
        return val
    if isinstance(val, (int, np.integer)):
        return Fraction(int(val), 1)
    return Fraction(float(val)).limit_denominator(10000)

def fmt_frac(val):
    if isinstance(val, str):
        return val
    try:
        if isinstance(val, float) and np.isinf(val):
            return "∞"
        f = to_frac(val)
        if abs(f) >= Fraction(100000, 1):
            return "-M" if f < 0 else "M"
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        return str(val)

def fmt_latex_frac(val, signed=False):
    f = to_frac(val)
    if f == 0:
        return "+ 0" if signed else "0"
    sign = ""
    if signed:
        sign = "+ " if f > 0 else "- "
    elif f < 0:
        sign = "-"
    af = abs(f)
    mag = str(af.numerator) if af.denominator == 1 else f"\\frac{{{af.numerator}}}{{{af.denominator}}}"
    return f"{sign}{mag}"

def frac_floor_decomp(f):
    f = to_frac(f)
    fl = math.floor(f)
    rem = f - fl
    return fl, rem

# ==========================================
# HELPER: TABLEAU BUILDER (EXACT FRACTIONS)
# ==========================================
def build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False):
    num_cons = len(basis_indices)
    cb = [cost_row[idx] for idx in basis_indices]

    cj_display = ["", "C_j ->", ""] + [fmt_frac(val) for val in cost_row]
    if include_ratio_col:
        cj_display += [""]

    rows = [cj_display]
    for i in range(num_cons):
        row_vals = [
            fmt_frac(cb[i]),
            var_names[basis_indices[i]],
            fmt_frac(rhs[i])
        ] + [fmt_frac(x) for x in tableau[i]]
        if include_ratio_col:
            r_val = ratios[i] if ratios is not None else "—"
            row_vals += [fmt_frac(r_val) if r_val != "—" else "—"]
        rows.append(row_vals)

    z_disp = fmt_frac(z_val if obj_type == "Maximize" else -z_val)
    zj_row = ["", "Z_j", z_disp] + [fmt_frac(val) for val in zj]
    if include_ratio_col:
        zj_row += [""]

    cj_zj_row = ["", "C_j - Z_j", ""] + [fmt_frac(val) for val in cj_zj]
    if include_ratio_col:
        cj_zj_row += [""]

    rows.append(zj_row)
    rows.append(cj_zj_row)

    if include_ratio_row and ratios is not None:
        ratio_row = ["", "Ratio", ""] + [fmt_frac(r) if r != "—" else "—" for r in ratios]
        if include_ratio_col:
            ratio_row += [""]
        rows.append(ratio_row)

    cols = ["C_B", "X_B", "b"] + var_names
    if include_ratio_col:
        cols += ["Ratio"]

    return pd.DataFrame(rows, columns=cols)

def get_point_from_basis(basis_indices, rhs, num_vars):
    pt = [Fraction(0, 1)] * num_vars
    for pos, b_idx in enumerate(basis_indices):
        if b_idx < num_vars:
            pt[b_idx] = rhs[pos]
    return tuple(pt[:2])

# ==========================================
# HELPER: EXACT FRACTION BIG-M SIMPLEX ENGINE
# ==========================================
def solve_simplex_big_m(obj_type, c, A, signs, b):
    M = Fraction(100000, 1)
    num_vars = len(c)
    num_cons = len(b)

    A_f = [[to_frac(A[i][j]) for j in range(num_vars)] for i in range(num_cons)]
    b_f = [to_frac(b[i]) for i in range(num_cons)]
    signs = list(signs)

    for i in range(num_cons):
        if b_f[i] < 0:
            b_f[i] = -b_f[i]
            A_f[i] = [-x for x in A_f[i]]
            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    c_eff = [to_frac(x) if obj_type == "Maximize" else -to_frac(x) for x in c]
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)

    extra_cols, basis_indices, artificial_indices = [], [], []
    s_count, a_count = 1, 1
    for i, s in enumerate(signs):
        if s == "<=":
            col = [Fraction(0, 1)] * num_cons
            col[i] = Fraction(1, 1)
            extra_cols.append(col)
            var_names.append(f"s{s_count}")
            cost_row.append(Fraction(0, 1))
            basis_indices.append(len(var_names) - 1)
            s_count += 1
        elif s == ">=":
            col_s = [Fraction(0, 1)] * num_cons
            col_s[i] = Fraction(-1, 1)
            extra_cols.append(col_s)
            var_names.append(f"s{s_count}")
            cost_row.append(Fraction(0, 1))
            s_count += 1

            col_a = [Fraction(0, 1)] * num_cons
            col_a[i] = Fraction(1, 1)
            extra_cols.append(col_a)
            var_names.append(f"A{a_count}")
            cost_row.append(-M)
            basis_indices.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)
            a_count += 1
        elif s == "=":
            col_a = [Fraction(0, 1)] * num_cons
            col_a[i] = Fraction(1, 1)
            extra_cols.append(col_a)
            var_names.append(f"A{a_count}")
            cost_row.append(-M)
            basis_indices.append(len(var_names) - 1)
            artificial_indices.append(len(var_names) - 1)
            a_count += 1

    tableau = []
    for i in range(num_cons):
        row = list(A_f[i]) + [col[i] for col in extra_cols]
        tableau.append(row)

    total_cols = len(var_names)
    rhs = list(b_f)
    iterations, special_flags = [], set()
    status = "Optimal"
    is_degenerate = False
    has_alternative = False
    alt_solutions = []
    degenerate_points = set()
    final_tableau_state = None

    for it in range(30):
        cb = [cost_row[idx] for idx in basis_indices]
        zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(total_cols)]
        cj_zj = [cost_row[j] - zj[j] for j in range(total_cols)]
        z_val = sum(cb[i] * rhs[i] for i in range(num_cons))

        for i, r_val in enumerate(rhs):
            if r_val == 0:
                is_degenerate = True
                deg_pt = get_point_from_basis(basis_indices, rhs, num_vars)
                degenerate_points.add(deg_pt)
                deg_str = f"({fmt_frac(deg_pt[0])}, {fmt_frac(deg_pt[1])})"
                special_flags.add(f"⚠️ **Degeneracy Detected:** Basic variable **{var_names[basis_indices[i]]}** has $b_i = 0$ at row {i+1} (Vertex: {deg_str}).")

        if max(cj_zj) <= 0:
            non_basic_indices = [j for j in range(total_cols) if j not in basis_indices]
            for nb_idx in non_basic_indices:
                if cj_zj[nb_idx] == 0:
                    has_alternative = True
                    special_flags.add(f"✨ **Alternative Optima Detected:** Non-basic variable **{var_names[nb_idx]}** has $C_j - Z_j = 0$.")
                    col_vals = [tableau[i][nb_idx] for i in range(num_cons)]
                    if any(v > 0 for v in col_vals):
                        valid_r = [(rhs[i] / col_vals[i], i) for i in range(num_cons) if col_vals[i] > 0]
                        if valid_r:
                            l_row = min(valid_r, key=lambda x: x[0])[1]
                            t_dup = [list(r) for r in tableau]
                            r_dup = list(rhs)
                            b_dup = list(basis_indices)
                            p_val = t_dup[l_row][nb_idx]
                            t_dup[l_row] = [x / p_val for x in t_dup[l_row]]
                            r_dup[l_row] /= p_val
                            for ri in range(num_cons):
                                if ri != l_row:
                                    fac = t_dup[ri][nb_idx]
                                    t_dup[ri] = [t_dup[ri][c_i] - fac * t_dup[l_row][c_i] for c_i in range(total_cols)]
                                    r_dup[ri] -= fac * r_dup[l_row]
                            b_dup[l_row] = nb_idx
                            alt_sol = [Fraction(0, 1)] * num_vars
                            for pos, b_idx in enumerate(b_dup):
                                if b_idx < num_vars:
                                    alt_sol[b_idx] = r_dup[pos]
                            alt_solutions.append(alt_sol)

            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False)
            iterations.append((it + 1, df, "Optimal Tableau Reached (All $C_j - Z_j \\le 0$)", list(basis_indices), list(rhs)))
            final_tableau_state = {
                "tableau": [list(r) for r in tableau],
                "rhs": list(rhs),
                "basis_indices": list(basis_indices),
                "var_names": list(var_names),
                "cost_row": list(cost_row),
                "zj": list(zj),
                "cj_zj": list(cj_zj),
                "z_val": z_val
            }
            break

        entering_col = max(range(total_cols), key=lambda j: cj_zj[j])
        col_vals = [tableau[i][entering_col] for i in range(num_cons)]

        if all(v <= 0 for v in col_vals):
            status = "Unbounded"
            special_flags.add(f"🚩 **Unboundedness:** Pivot column {var_names[entering_col]} has entries ≤ 0.")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=True, include_ratio_row=False)
            iterations.append((it + 1, df, f"Unbounded at {var_names[entering_col]}", list(basis_indices), list(rhs)))
            break

        valid_ratios = []
        ratio_display = ["—"] * num_cons
        for i in range(num_cons):
            if col_vals[i] > 0:
                r_val = rhs[i] / col_vals[i]
                valid_ratios.append((r_val, i))
                ratio_display[i] = r_val
            else:
                ratio_display[i] = "—"

        leaving_row = min(valid_ratios, key=lambda x: x[0])[1]
        df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=ratio_display, include_ratio_col=True, include_ratio_row=False)
        iterations.append((it + 1, df, f"Entering: **{var_names[entering_col]}** | Leaving: **{var_names[basis_indices[leaving_row]]}**", list(basis_indices), list(rhs)))

        pivot_val = tableau[leaving_row][entering_col]
        tableau[leaving_row] = [x / pivot_val for x in tableau[leaving_row]]
        rhs[leaving_row] /= pivot_val
        for i in range(num_cons):
            if i != leaving_row:
                factor = tableau[i][entering_col]
                tableau[i] = [tableau[i][c_i] - factor * tableau[leaving_row][c_i] for c_i in range(total_cols)]
                rhs[i] -= factor * rhs[leaving_row]
        basis_indices[leaving_row] = entering_col

    sol = [Fraction(0, 1)] * num_vars
    for pos, b_idx in enumerate(basis_indices):
        if b_idx < num_vars:
            sol[b_idx] = rhs[pos]
    c_orig = [to_frac(x) for x in c]
    final_z = sum(c_orig[j] * sol[j] for j in range(num_vars))
    return status, sol, final_z, iterations, list(special_flags), final_tableau_state

# ==========================================
# HELPER: EXACT FRACTION DUAL SIMPLEX ENGINE
# ==========================================
def run_dual_simplex_on_tableau(obj_type, tableau, var_names, cost_row, basis_indices, rhs, num_vars):
    tableau = [list(r) for r in tableau]
    rhs = list(rhs)
    basis_indices = list(basis_indices)
    var_names = list(var_names)
    cost_row = list(cost_row)
    num_cons = len(basis_indices)
    total_cols = len(var_names)

    iterations, special_flags = [], set()
    status = "Optimal"
    final_state = None

    for it in range(30):
        cb = [cost_row[idx] for idx in basis_indices]
        zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(total_cols)]
        cj_zj = [cost_row[j] - zj[j] for j in range(total_cols)]
        z_val = sum(cb[i] * rhs[i] for i in range(num_cons))

        if max(cj_zj) > 0:
            status = "Dual Infeasible"
            special_flags.add("🚩 **Dual Infeasibility:** $C_j - Z_j > 0$ found.")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=False, include_ratio_row=True)
            iterations.append((it + 1, df, "Dual Infeasible Tableau"))
            break

        if all(r >= 0 for r in rhs):
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=False, include_ratio_row=True)
            iterations.append((it + 1, df, "Optimal Dual Simplex Tableau Reached (All $b_i \\ge 0$)"))
            final_state = {
                "tableau": [list(r) for r in tableau],
                "rhs": list(rhs),
                "basis_indices": list(basis_indices),
                "var_names": list(var_names),
                "cost_row": list(cost_row),
                "zj": list(zj),
                "cj_zj": list(cj_zj),
                "z_val": z_val
            }
            break

        leaving_row = min(range(num_cons), key=lambda i: rhs[i])
        if rhs[leaving_row] >= 0:
            break

        row_vals = tableau[leaving_row]
        if all(v >= 0 for v in row_vals):
            status = "Infeasible"
            special_flags.add(f"🚩 **Primal Infeasibility / Dual Unboundedness:** Leaving row {leaving_row+1} has all entries $\\ge 0$ with negative RHS.")
            df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=None, include_ratio_col=False, include_ratio_row=True)
            iterations.append((it + 1, df, f"Infeasible at Row {leaving_row+1}"))
            break

        valid_ratios = []
        ratio_display = ["—"] * total_cols
        for j in range(total_cols):
            if row_vals[j] < 0:
                r_val = abs(cj_zj[j] / row_vals[j])
                valid_ratios.append((r_val, j))
                ratio_display[j] = r_val
            else:
                ratio_display[j] = "—"

        if not valid_ratios:
            status = "Infeasible"
            break

        entering_col = min(valid_ratios, key=lambda x: x[0])[1]
        df = build_notebook_tableau(tableau, var_names, cost_row, basis_indices, rhs, zj, cj_zj, z_val, obj_type, ratios=ratio_display, include_ratio_col=False, include_ratio_row=True)
        iter_note = f"Leaving: **{var_names[basis_indices[leaving_row]]}** | Entering: **{var_names[entering_col]}**"
        iterations.append((it + 1, df, iter_note))

        pivot_val = tableau[leaving_row][entering_col]
        tableau[leaving_row] = [x / pivot_val for x in tableau[leaving_row]]
        rhs[leaving_row] /= pivot_val
        for i in range(num_cons):
            if i != leaving_row:
                factor = tableau[i][entering_col]
                tableau[i] = [tableau[i][c_i] - factor * tableau[leaving_row][c_i] for c_i in range(total_cols)]
                rhs[i] -= factor * rhs[leaving_row]
        basis_indices[leaving_row] = entering_col

    sol = [Fraction(0, 1)] * num_vars
    for pos, b_idx in enumerate(basis_indices):
        if b_idx < num_vars:
            sol[b_idx] = rhs[pos]

    c_orig = [cost_row[j] if obj_type == "Maximize" else -cost_row[j] for j in range(num_vars)]
    final_z_val = sum(c_orig[j] * sol[j] for j in range(num_vars))
    return status, sol, final_z_val, iterations, list(special_flags), final_state

def solve_simplex_dual(obj_type, c, A, signs, b):
    num_vars = len(c)
    num_cons = len(b)

    A_f = [[to_frac(A[i][j]) for j in range(num_vars)] for i in range(num_cons)]
    b_f = [to_frac(b[i]) for i in range(num_cons)]
    signs = list(signs)

    for i in range(num_cons):
        if signs[i] == ">=":
            A_f[i] = [-x for x in A_f[i]]
            b_f[i] = -b_f[i]
            signs[i] = "<="

    c_eff = [to_frac(x) if obj_type == "Maximize" else -to_frac(x) for x in c]
    var_names = [f"x{j+1}" for j in range(num_vars)]
    cost_row = list(c_eff)

    extra_cols, basis_indices = [], []
    s_count = 1
    for i, s in enumerate(signs):
        col = [Fraction(0, 1)] * num_cons
        col[i] = Fraction(1, 1)
        extra_cols.append(col)
        var_names.append(f"s{s_count}")
        cost_row.append(Fraction(0, 1))
        basis_indices.append(len(var_names) - 1)
        s_count += 1

    tableau = [list(A_f[i]) + [col[i] for col in extra_cols] for i in range(num_cons)]
    status, sol, final_z_val, iterations, flags, _ = run_dual_simplex_on_tableau(
        obj_type, tableau, var_names, cost_row, basis_indices, b_f, num_vars
    )
    return status, sol, final_z_val, iterations, flags

# ==========================================
# HELPER: GOMORY FRACTIONAL CUT GENERATOR
# ==========================================
def generate_gomory_cut_from_row(tableau, rhs, basis_indices, var_names, cost_row, signs, A, b, row_idx, cut_num, num_vars, cut_stages_so_far):
    total_cols = len(var_names)
    row_coeffs = tableau[row_idx]
    rhs_val = rhs[row_idx]
    b_var_name = var_names[basis_indices[row_idx]]

    lhs_terms = []
    decomp_terms = []
    cut_lhs_terms = []
    cut_row_coeffs = []

    for j in range(total_cols):
        coeff = row_coeffs[j]
        vname = var_names[j]
        if coeff != 0:
            if not lhs_terms:
                lhs_terms.append(f"{fmt_latex_frac(coeff)} {vname}" if coeff != 1 else vname)
            else:
                if coeff == 1:
                    lhs_terms.append(f"+ {vname}")
                elif coeff == -1:
                    lhs_terms.append(f"- {vname}")
                else:
                    lhs_terms.append(f"{fmt_latex_frac(coeff, signed=True)} {vname}")

        if j == basis_indices[row_idx]:
            decomp_terms.append(vname)
            cut_row_coeffs.append(Fraction(0, 1))
        else:
            fl, rem = frac_floor_decomp(coeff)
            cut_row_coeffs.append(-rem)
            if coeff != 0:
                if fl == 0:
                    decomp_terms.append(f"{fmt_latex_frac(rem, signed=True)} {vname}")
                elif rem == 0:
                    decomp_terms.append(f"{fmt_latex_frac(fl, signed=True)} {vname}")
                else:
                    decomp_terms.append(f"+ \\left({fmt_latex_frac(fl)} {fmt_latex_frac(rem, signed=True)}\\right) {vname}")
            if rem > 0:
                cut_lhs_terms.append(f"{fmt_latex_frac(-rem, signed=bool(cut_lhs_terms))} {vname}")

    rhs_fl, rhs_rem = frac_floor_decomp(rhs_val)
    eq_source = f"{b_var_name}\\text{{-row}} \\implies {' '.join(lhs_terms)} = {fmt_latex_frac(rhs_val)}"
    eq_decomp = f"{' '.join(decomp_terms)} = {rhs_fl} + {fmt_latex_frac(rhs_rem)}"

    cut_slack_name = f"S_{cut_num}"
    cut_slack_plain = f"S{cut_num}"
    cut_rhs = -rhs_rem
    eq_cut = f"\\text{{Gomory cut: }} {' '.join(cut_lhs_terms)} + {cut_slack_name} = {fmt_latex_frac(cut_rhs)}"

    orig_x_coeffs = [Fraction(0, 1)] * num_vars
    orig_rhs_const = rhs_rem
    for j in range(total_cols):
        if j == basis_indices[row_idx]:
            continue
        _, rem = frac_floor_decomp(row_coeffs[j])
        if rem == 0:
            continue
        if j < num_vars:
            orig_x_coeffs[j] += rem
        else:
            vname = var_names[j]
            if vname.startswith("s") and vname[1:].isdigit():
                c_idx = int(vname[1:]) - 1
                if 0 <= c_idx < len(b):
                    if signs[c_idx] == "<=":
                        orig_rhs_const -= rem * to_frac(b[c_idx])
                        for k in range(num_vars):
                            orig_x_coeffs[k] -= rem * to_frac(A[c_idx][k])
                    elif signs[c_idx] == ">=":
                        orig_rhs_const += rem * to_frac(b[c_idx])
                        for k in range(num_vars):
                            orig_x_coeffs[k] += rem * to_frac(A[c_idx][k])
            elif vname.startswith("S") and vname[1:].isdigit():
                cut_m_idx = int(vname[1:]) - 1
                if 0 <= cut_m_idx < len(cut_stages_so_far):
                    prev_cut = cut_stages_so_far[cut_m_idx]
                    orig_rhs_const += rem * prev_cut["orig_rhs_const"]
                    for k in range(num_vars):
                        orig_x_coeffs[k] += rem * prev_cut["orig_x_coeffs"][k]

    return {
        "row_idx": row_idx,
        "b_var_name": b_var_name,
        "rhs_val": rhs_val,
        "eq_source": eq_source,
        "eq_decomp": eq_decomp,
        "eq_cut": eq_cut,
        "cut_row_coeffs": cut_row_coeffs,
        "cut_slack_plain": cut_slack_plain,
        "cut_rhs": cut_rhs,
        "orig_x_coeffs": orig_x_coeffs,
        "orig_rhs_const": orig_rhs_const
    }

# ==========================================
# HELPER: 2D FEASIBLE POLYGON GENERATOR
# ==========================================
def get_2d_feasible_polygon(A_rows, signs, b_vals, max_bound):
    num_cons = len(b_vals)
    candidates = [(0.0, 0.0), (0.0, max_bound), (max_bound, 0.0), (max_bound, max_bound)]

    for i in range(num_cons):
        a1, a2, b_i = A_rows[i][0], A_rows[i][1], b_vals[i]
        if abs(a2) > 1e-7:
            candidates.append((0.0, b_i / a2))
        if abs(a1) > 1e-7:
            candidates.append((b_i / a1, 0.0))
        if abs(a2) > 1e-7:
            candidates.append((max_bound, (b_i - a1 * max_bound) / a2))
        if abs(a1) > 1e-7:
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
        if x < -1e-5 or y < -1e-5 or x > max_bound + 1e-5 or y > max_bound + 1e-5:
            continue
        ok = True
        for i in range(num_cons):
            val = A_rows[i][0] * x + A_rows[i][1] * y
            s = signs[i]
            b_i = b_vals[i]
            if s == "<=" and val > b_i + 1e-3:
                ok = False
                break
            elif s == ">=" and val < b_i - 1e-3:
                ok = False
                break
            elif s == "=" and not np.isclose(val, b_i, atol=1e-2):
                ok = False
                break
        if ok:
            pt_clean = (round(max(0.0, x), 4), round(max(0.0, y), 4))
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

    row1_c1, row1_c2, row1_c3 = st.columns(3, gap="large")
    with row1_c1: st.button("📈 Simplex Method", on_click=select_problem, args=("Simplex Method",))
    with row1_c2: st.button("✂️ Cutting Plane Method", on_click=select_problem, args=("Cutting Plane Method",))
    with row1_c3: st.button("🔄 Dual Simplex Method", on_click=select_problem, args=("Dual Simplex Method",))

    st.write("")
    row2_space1, row2_c1, row2_c2, row2_space2 = st.columns([0.5, 3, 3, 0.5], gap="large")
    with row2_c1: st.button("🌳 Branch and Bound", on_click=select_problem, args=("Branch and Bound",))
    with row2_c2: st.button("🌐 Network & TSP Solver", on_click=select_problem, args=("Network & TSP Solver",))

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
    # 2. CUTTING PLANE METHOD (Interactive & Cumulative Multi-Cut Chain)
    # ------------------------------------------
    if problem == "Cutting Plane Method":
        st.markdown("### ✂️ Cutting Plane Method (Interactive Gomory Fractional Cut Studio)")
        st.markdown("Solve Integer Linear Programming (ILP) problems interactively by selecting rows with fractional right-hand sides to generate Gomory cuts step-by-step.")

        cp_cfg1, cp_cfg2 = st.columns(2)
        with cp_cfg1: cp_obj_type = st.selectbox("Objective Goal", ["Maximize", "Minimize"], key="cp_obj")
        with cp_cfg2: cp_num_cons = st.number_input("Constraints count", min_value=1, max_value=10, value=2, step=1, key="cp_c")

        st.markdown("#### Objective Function Coefficients ($c_j$ for $x_1, x_2$):")
        cp_c_cols = st.columns(2)
        cp_c_vals = []
        with cp_c_cols[0]: cp_c_vals.append(st.number_input("c1", value=1.0, key="cp_c1"))
        with cp_c_cols[1]: cp_c_vals.append(st.number_input("c2", value=1.0, key="cp_c2"))

        st.markdown("#### Constraints Setup:")
        cp_A, cp_signs, cp_b = [], [], []
        default_cp_a = [[2.0, 1.0], [1.0, 2.0]]
        default_cp_b = [4.0, 4.0]
        for i in range(int(cp_num_cons)):
            st.markdown(f"**Constraint {i+1}:**")
            r_cols = st.columns(4)
            r_a = []
            for j in range(2):
                def_val = default_cp_a[i][j] if i < 2 else 1.0
                with r_cols[j]: r_a.append(st.number_input(f"cp_a_{i}_{j}", value=def_val, key=f"cp_a_{i}_{j}"))
            with r_cols[2]: cp_signs.append(st.selectbox(f"cp_sign_{i}", ["<=", ">=", "="], key=f"cp_sign_{i}"))
            with r_cols[3]:
                def_b = default_cp_b[i] if i < 2 else 5.0
                cp_b.append(st.number_input(f"cp_b_{i}", value=def_b, key=f"cp_b_{i}"))
            cp_A.append(r_a)

        if "cp_initialized" not in st.session_state:
            st.session_state.cp_initialized = False
            st.session_state.cp_cut_stages = []

        if st.button("🚀 Initialize LP Relaxation & Solve Continuous Optimum"):
            st.session_state.cp_cut_stages = []
            res_init = solve_simplex_big_m(cp_obj_type, np.array(cp_c_vals), np.array(cp_A), cp_signs, np.array(cp_b))
            st.session_state.cp_status = res_init[0]
            st.session_state.cp_sol = res_init[1]
            st.session_state.cp_z = res_init[2]
            st.session_state.cp_initial_iterations = res_init[3]
            st.session_state.cp_state = res_init[5]
            st.session_state.cp_initialized = True
            st.rerun()

        if st.session_state.cp_initialized:
            st.divider()
            st.markdown("### 📋 Continuous LP Relaxation Optimal Tableau")
            if st.session_state.cp_initial_iterations:
                opt_iter_num, opt_df, opt_note, opt_basis, opt_rhs = st.session_state.cp_initial_iterations[-1]
                st.markdown(f"**{opt_note}**")
                st.dataframe(opt_df, use_container_width=True, hide_index=True)

            # Display all previously applied cut stages in sequence
            if st.session_state.cp_cut_stages:
                for stage in st.session_state.cp_cut_stages:
                    st.divider()
                    st.markdown(f"### ✂️ Gomory Fractional Cut #{stage['cut_num']} & Dual Simplex Solution")
                    st.latex(stage["eq_source"])
                    st.latex(stage["eq_decomp"])
                    st.latex(stage["eq_cut"])

                    for d_it_num, d_df, d_note in stage["dual_iters"]:
                        st.markdown(f"**Dual Simplex Iteration {d_it_num} (Cut #{stage['cut_num']})** — {d_note}")
                        st.dataframe(d_df, use_container_width=True, hide_index=True)

                    if stage["dual_flags"]:
                        for flg in stage["dual_flags"]:
                            st.warning(flg)

                    if stage["dual_status"] == "Optimal":
                        d_sol = stage["dual_sol"]
                        st.info(f"**Post-Cut #{stage['cut_num']} Basic Solution:** **x₁ = {fmt_frac(d_sol[0])}**, **x₂ = {fmt_frac(d_sol[1])}** | **Z = {fmt_frac(stage['dual_z'])}**")

            curr_sol = st.session_state.cp_sol
            is_all_integer = all(v.denominator == 1 for v in curr_sol)

            if is_all_integer:
                st.success(f"🏆 **Optimal Integer Solution Reached:** **x₁ = {fmt_frac(curr_sol[0])}**, **x₂ = {fmt_frac(curr_sol[1])}** | **Z* = {fmt_frac(st.session_state.cp_z)}**")
            else:
                st.info(f"Current Solution is Fractional: **x₁ = {fmt_frac(curr_sol[0])}**, **x₂ = {fmt_frac(curr_sol[1])}**. Select a row below to generate the next Gomory Cut.")

                candidate_rows = []
                curr_state = st.session_state.cp_state
                if curr_state is not None:
                    tableau = curr_state["tableau"]
                    rhs = curr_state["rhs"]
                    basis_indices = curr_state["basis_indices"]
                    var_names = curr_state["var_names"]
                    cost_row = curr_state["cost_row"]

                    for pos, b_idx in enumerate(basis_indices):
                        rhs_val = rhs[pos]
                        fl, rem = frac_floor_decomp(rhs_val)
                        if rem > 0:
                            b_var_name = var_names[b_idx]
                            preview_cut = generate_gomory_cut_from_row(
                                tableau, rhs, basis_indices, var_names, cost_row,
                                cp_signs, cp_A, cp_b, pos, len(st.session_state.cp_cut_stages) + 1, 2,
                                st.session_state.cp_cut_stages
                            )
                            ca1 = preview_cut["orig_x_coeffs"][0]
                            ca2 = preview_cut["orig_x_coeffs"][1]
                            cb_r = preview_cut["orig_rhs_const"]
                            disp_label = f"Cut from Row {b_var_name} (RHS = {fmt_frac(rhs_val)}) → ({fmt_frac(ca1)})x₁ + ({fmt_frac(ca2)})x₂ ≥ {fmt_frac(cb_r)}"
                            candidate_rows.append((pos, disp_label))

                    if candidate_rows:
                        selected_option = st.selectbox(
                            f"Select Cutting Plane #{len(st.session_state.cp_cut_stages)+1}:",
                            options=candidate_rows,
                            format_func=lambda x: x[1],
                            key=f"cp_select_stage_{len(st.session_state.cp_cut_stages)}"
                        )
                        selected_pos = selected_option[0]

                        if st.button("✂️ Apply Selected Cut", key=f"apply_cut_btn_stage_{len(st.session_state.cp_cut_stages)}"):
                            cut_num = len(st.session_state.cp_cut_stages) + 1
                            cut_info = generate_gomory_cut_from_row(
                                tableau, rhs, basis_indices, var_names, cost_row,
                                cp_signs, cp_A, cp_b, selected_pos, cut_num, 2,
                                st.session_state.cp_cut_stages
                            )

                            new_tableau = [row + [Fraction(0, 1)] for row in tableau]
                            new_cut_row = cut_info["cut_row_coeffs"] + [Fraction(1, 1)]
                            new_tableau.append(new_cut_row)

                            new_var_names = var_names + [cut_info["cut_slack_plain"]]
                            new_cost_row = cost_row + [Fraction(0, 1)]
                            new_basis_indices = basis_indices + [len(new_var_names) - 1]
                            new_rhs = rhs + [cut_info["cut_rhs"]]

                            d_status, d_sol, d_z, d_iters, d_flags, d_state = run_dual_simplex_on_tableau(
                                cp_obj_type, new_tableau, new_var_names, new_cost_row, new_basis_indices, new_rhs, 2
                            )

                            stage_data = {
                                "cut_num": cut_num,
                                "source_var": cut_info["b_var_name"],
                                "eq_source": cut_info["eq_source"],
                                "eq_decomp": cut_info["eq_decomp"],
                                "eq_cut": cut_info["eq_cut"],
                                "dual_status": d_status,
                                "dual_sol": d_sol,
                                "dual_z": d_z,
                                "dual_iters": d_iters,
                                "dual_flags": d_flags,
                                "orig_x_coeffs": cut_info["orig_x_coeffs"],
                                "orig_rhs_const": cut_info["orig_rhs_const"]
                            }

                            st.session_state.cp_cut_stages.append(stage_data)
                            if d_state is not None:
                                st.session_state.cp_state = d_state
                                st.session_state.cp_status = d_status
                                st.session_state.cp_sol = d_sol
                                st.session_state.cp_z = d_z
                            st.rerun()

            # Geometrical Visualization with ALL Cumulative Cuts Visible
            st.divider()
            st.markdown("### 📈 Cutting Plane Feasible Region & Cumulative Cut Slicing")
            st.caption("Visualizing initial constraints, integer lattice points (dots), continuous LP optimal point, and **all applied Gomory cuts** (dashed lines).")

            max_rhs_cp = max([abs(val) for val in cp_b]) if cp_b else 10.0
            max_bound_cp = max(max_rhs_cp * 1.5, 6.0)
            x_line_cp = np.linspace(0, max_bound_cp, 200)

            fig_cp = go.Figure()
            colors_cp = ['#38bdf8', '#f472b6', '#34d399', '#fbbf24', '#a78bfa']

            initial_verts = get_2d_feasible_polygon(cp_A, cp_signs, cp_b, max_bound_cp)
            if len(initial_verts) >= 3:
                px = [p[0] for p in initial_verts] + [initial_verts[0][0]]
                py = [p[1] for p in initial_verts] + [initial_verts[0][1]]
                fig_cp.add_trace(go.Scatter(x=px, y=py, fill="toself", fillcolor="rgba(56, 189, 248, 0.2)", line=dict(color="#38bdf8", width=2.5), name="Initial Feasible Region"))

            for i in range(len(cp_b)):
                a1, a2, b_i = cp_A[i][0], cp_A[i][1], cp_b[i]
                col_c = colors_cp[i % len(colors_cp)]
                if abs(a2) > 1e-6:
                    y_vals = (b_i - a1 * x_line_cp) / a2
                    mask = (y_vals >= 0) & (y_vals <= max_bound_cp)
                    if np.any(mask):
                        fig_cp.add_trace(go.Scatter(x=x_line_cp[mask], y=y_vals[mask], mode="lines", name=f"C{i+1}: {fmt_frac(a1)}x₁ + {fmt_frac(a2)}x₂ {cp_signs[i]} {fmt_frac(b_i)}", line=dict(color=col_c, width=2.5)))

            lattice_x, lattice_y = [], []
            for ix in range(int(max_bound_cp) + 1):
                for iy in range(int(max_bound_cp) + 1):
                    feasible = True
                    for i in range(len(cp_b)):
                        val = cp_A[i][0] * ix + cp_A[i][1] * iy
                        if cp_signs[i] == "<=" and val > cp_b[i] + 1e-3:
                            feasible = False
                            break
                        elif cp_signs[i] == ">=" and val < cp_b[i] - 1e-3:
                            feasible = False
                            break
                    if feasible:
                        lattice_x.append(ix)
                        lattice_y.append(iy)

            fig_cp.add_trace(go.Scatter(
                x=lattice_x, y=lattice_y,
                mode="markers",
                marker=dict(size=8, color="#94a3b8", symbol="circle"),
                name="Integer Lattice Points"
            ))

            cut_colors = ['#f43f5e', '#a855f7', '#fbbf24', '#34d399', '#38bdf8']
            for stage in st.session_state.cp_cut_stages:
                ca1 = float(stage["orig_x_coeffs"][0])
                ca2 = float(stage["orig_x_coeffs"][1])
                cb_r = float(stage["orig_rhs_const"])
                c_num = stage["cut_num"]
                c_col = cut_colors[(c_num - 1) % len(cut_colors)]

                if abs(ca2) > 1e-6:
                    y_cut = (cb_r - ca1 * x_line_cp) / ca2
                    mask_c = (y_cut >= 0) & (y_cut <= max_bound_cp)
                    if np.any(mask_c):
                        fig_cp.add_trace(go.Scatter(
                            x=x_line_cp[mask_c], y=y_cut[mask_c],
                            mode="lines",
                            line=dict(color=c_col, width=3, dash="dash"),
                            name=f"Gomory Cut #{c_num} ({fmt_frac(ca1)}x₁ + {fmt_frac(ca2)}x₂ ≥ {fmt_frac(cb_r)})"
                        ))
                elif abs(ca1) > 1e-6:
                    x_cut = cb_r / ca1
                    if 0 <= x_cut <= max_bound_cp:
                        fig_cp.add_trace(go.Scatter(
                            x=[x_cut, x_cut], y=[0, max_bound_cp],
                            mode="lines",
                            line=dict(color=c_col, width=3, dash="dash"),
                            name=f"Gomory Cut #{c_num} (x₁ ≥ {fmt_frac(x_cut)})"
                        ))

            fig_cp.update_layout(
                xaxis=dict(title=dict(text="Decision Variable x₁", font=dict(color="#f8fafc", size=14)), range=[0, max_bound_cp], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                yaxis=dict(title=dict(text="Decision Variable x₂", font=dict(color="#f8fafc", size=14)), range=[0, max_bound_cp], dtick=1, gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                legend=dict(font=dict(color="#f8fafc", size=12), bgcolor="rgba(30, 41, 59, 0.8)", bordercolor="#334155"),
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=580
            )
            st.plotly_chart(fig_cp, use_container_width=True)
