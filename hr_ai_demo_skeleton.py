from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ortools.linear_solver import pywraplp
except Exception:
    pywraplp = None

# =============================
# Data Classes
# =============================

@dataclass
class Employee:
    employee_id: str
    full_name: str
    role: str
    home_store_id: str
    performance_score: float
    feedback_score: float
    attendance_score: float

    @property
    def talent_score(self) -> float:
        return 0.5 * self.performance_score + 0.3 * self.feedback_score + 0.2 * self.attendance_score

@dataclass
class Store:
    store_id: str
    store_name: str
    lat: float
    lon: float

# =============================
# Helpers
# =============================

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# =============================
# Load Data
# =============================

def load_csvs(data_dir: str) -> Dict[str, pd.DataFrame]:
    req = ["employees.csv", "stores.csv", "shift_demands.csv", "assignments.csv", "leaves.csv"]
    dfs: Dict[str, pd.DataFrame] = {}
    for f in req:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
        dfs[f.split(".")[0]] = pd.read_csv(p)
    # distances optional
    dist_path = os.path.join(data_dir, "distances.csv")
    dfs["distances"] = pd.read_csv(dist_path) if os.path.exists(dist_path) else None
    return dfs

def compute_distance_matrix(stores: pd.DataFrame, distances: pd.DataFrame | None) -> pd.DataFrame:
    if distances is not None:
        return distances.copy()
    pairs = []
    for s1 in stores.itertuples(index=False):
        for s2 in stores.itertuples(index=False):
            pairs.append({
                "from_store_id": s1.store_id,
                "to_store_id": s2.store_id,
                "distance_km": haversine_km(s1.lat, s1.lon, s2.lat, s2.lon)
            })
    return pd.DataFrame(pairs)

# =============================
# Preprocess
# =============================

def expand_leaves(leaves: pd.DataFrame) -> pd.DataFrame:
    leaves = leaves.copy()
    leaves["start_date"] = pd.to_datetime(leaves["start_date"]).dt.date
    leaves["end_date"] = pd.to_datetime(leaves["end_date"]).dt.date
    rows = []
    for r in leaves.itertuples(index=False):
        for d in daterange(r.start_date, r.end_date):
            rows.append({"employee_id": r.employee_id, "date": d})
    return pd.DataFrame(rows)

def compute_effective_supply(assignments: pd.DataFrame, leave_days: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    assn = assignments.copy()
    assn["date"] = pd.to_datetime(assn["date"]).dt.date
    assn = assn[(assn["date"] >= start) & (assn["date"] <= end)]
    # remove leave
    assn = assn.merge(leave_days, on=["employee_id", "date"], how="left", indicator=True)
    assn = assn[assn["_merge"] == "left_only"]
    assn["available"] = 1
    supply = assn.groupby(["store_id", "date", "role"], as_index=False)["available"].sum()
    supply = supply.rename(columns={"available": "available_count"})
    return supply

def compute_gaps(shift_demands: pd.DataFrame, supply: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    dem = shift_demands.copy()
    dem["date"] = pd.to_datetime(dem["date"]).dt.date
    dem = dem[(dem["date"] >= start) & (dem["date"] <= end)]
    df = dem.merge(supply, on=["store_id", "date", "role"], how="left")
    df["available_count"] = df["available_count"].fillna(0)
    df["gap"] = (df["required_count"] - df["available_count"]).astype(int)
    return df

def score_employees(employees: pd.DataFrame) -> pd.DataFrame:
    emp = employees.copy()
    for c in ["performance_score", "feedback_score", "attendance_score"]:
        emp[c] = emp[c].clip(0, 1)
    emp["talent_score"] = 0.5*emp["performance_score"] + 0.3*emp["feedback_score"] + 0.2*emp["attendance_score"]
    if emp["talent_score"].std(ddof=0) > 0:
        emp["talent_score_norm"] = (emp["talent_score"] - emp["talent_score"].min()) / (emp["talent_score"].max() - emp["talent_score"].min())
    else:
        emp["talent_score_norm"] = 0.5
    return emp

# =============================
# Optimization
# =============================

def optimize_reassignments(emp_scored: pd.DataFrame, gaps: pd.DataFrame, stores: pd.DataFrame, distances: pd.DataFrame, date_focus: date) -> pd.DataFrame:
    if pywraplp is None:
        raise RuntimeError("ortools not installed. Run: pip install ortools")

    G = gaps[(gaps["date"] == date_focus) & (gaps["gap"] > 0)].copy()
    if G.empty:
        return pd.DataFrame()

    E = emp_scored.copy()
    dist_lut = distances.set_index(["from_store_id", "to_store_id"])
    store_home = E.set_index("employee_id")["home_store_id"].to_dict()

    solver = pywraplp.Solver.CreateSolver("CBC")
    x = {}
    for e in E.itertuples(index=False):
        for g in G.itertuples(index=False):
            dkm = float(dist_lut.loc[(e.home_store_id, g.store_id)]["distance_km"]) if (e.home_store_id, g.store_id) in dist_lut.index else 1e6
            if dkm > 30:  # cap
                continue
            role_match = 1 if e.role == g.role else 0
            cost = dkm - 0.5*e.talent_score_norm + 5*(1-role_match)
            x[(e.employee_id, g.store_id, g.role)] = solver.IntVar(0, 1, f"x_{e.employee_id}_{g.store_id}_{g.role}")
            solver.Objective().SetCoefficient(x[(e.employee_id, g.store_id, g.role)], cost)

    if not x:
        return pd.DataFrame()

    solver.Objective().SetMinimization()

    # constraints
    by_emp: Dict[str, List] = {}
    for key, var in x.items():
        e_id, _, _ = key
        by_emp.setdefault(e_id, []).append(var)
    for e_id, vars_ in by_emp.items():
        c = solver.Constraint(0, 1)
        for v in vars_:
            c.SetCoefficient(v, 1)

    for (s, r), grp in G.groupby(["store_id", "role"]):
        c = solver.Constraint(0, int(grp["gap"].sum()))
        for key, var in x.items():
            if key[1] == s and key[2] == r:
                c.SetCoefficient(var, 1)

    solver.Solve()

    rows = []
    for key, var in x.items():
        if var.solution_value() > 0.5:
            e_id, to_store, role = key
            from_store = store_home[e_id]
            dkm = float(dist_lut.loc[(from_store, to_store)]["distance_km"])
            tscore = float(E.loc[E.employee_id == e_id, "talent_score"].iloc[0])
            rows.append({
                "employee_id": e_id,
                "from_store_id": from_store,
                "to_store_id": to_store,
                "role": role,
                "date": date_focus,
                "distance_km": round(dkm, 2),
                "talent_score": round(tscore, 3)
            })
    return pd.DataFrame(rows)

# =============================
# Pipeline
# =============================

def run_pipeline(data_dir: str, start: date, end: date, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dfs = load_csvs(data_dir)
    employees, stores, shift_demands, assignments, leaves = dfs["employees"], dfs["stores"], dfs["shift_demands"], dfs["assignments"], dfs["leaves"]
    distances = compute_distance_matrix(stores, dfs["distances"])

    leave_days = expand_leaves(leaves)
    supply = compute_effective_supply(assignments, leave_days, start, end)
    gaps = compute_gaps(shift_demands, supply, start, end)
    emp_scored = score_employees(employees)

    moves_all = []
    for d in daterange(start, end):
        m = optimize_reassignments(emp_scored, gaps, stores, distances, d)
        if not m.empty:
            moves_all.append(m)
    moves = pd.concat(moves_all, ignore_index=True) if moves_all else pd.DataFrame()

    gaps.to_csv(os.path.join(out_dir, "gaps.csv"), index=False)
    emp_scored.to_csv(os.path.join(out_dir, "employee_scores.csv"), index=False)
    moves.to_csv(os.path.join(out_dir, "reassignments.csv"), index=False)

    print("=== DEMO SUMMARY ===")
    print(f"Window {start} -> {end}")
    print(f"Total positive gaps: {int(gaps[gaps.gap>0]['gap'].sum())}")
    print(f"Moves suggested: {len(moves)}")
    if not moves.empty:
        print(moves.head().to_string(index=False))
    print(f"Outputs saved to {out_dir}")

# =============================
# CLI
# =============================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", dest="out_dir", default="./out")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    run_pipeline(args.data_dir, start, end, args.out_dir)
