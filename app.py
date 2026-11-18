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
    final_tableau_state = None

    for it in range(30):
        cb = [cost_row[idx] for idx in basis_indices]
        zj = [sum(cb[i] * tableau[i][j] for i in range(num_cons)) for j in range(total_cols)]
        cj_zj = [cost_row[j] - zj[j] for j in range(total_cols)]
        z_val = sum(cb[i] * rhs[i] for i in range(num_cons))

        if max(cj_zj) <= 0:
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

# ==========================================
# HELPER: GOMORY FRACTIONAL CUT GENERATOR
# ==========================================
def generate_gomory_cut_from_row(tableau, rhs, basis_indices, var_names, cost_row, signs, A, b, row_idx, cut_num, num_vars):
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
