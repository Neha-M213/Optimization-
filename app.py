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
