"""
attempt2/plots.py
=================
All plot generation for the attempt2 experiment.

Plot filenames use the convention:
    {metric}_vs_{axis}_alpha{value}_{dataset}[_h{hvalue}][_k{kvalue}].png

Examples:
    variance_vs_timestep_alpha03_HICHBA_h06_k40.png
    variance_vs_seedset_alpha03_HICHBA_h06.png
    reach_vs_seedset_alpha03_HICHBA_h06.png
    disparity_vs_seedset_alpha03_HICHBA_h06.png
    variance_vs_homophily_alpha03_HICHBA_k40.png
    dashboard_variance_alpha03_HICHBA.png
    heatmap_reduction_alpha03_HICHBA.png
    group_reach_vs_timestep_alpha03_HICHBA_h06_k40.png
    time_vs_k_variance_alpha03_HICHBA_h06_FGD.png
"""

import os
import sys

# Ensure the parent directory is in sys.path so 'attempt2' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Style constants ──────────────────────────────────────────────────────────

ALGO_STYLE = {
    # --- Ours ---
    "MYOPIC_HYBRID":       {"color": "#9C27B0", "marker": "*", "ls": "-",  "lw": 2.8,
                            "label": "Myopic-Hybrid (ours)"},
    "HEUR2":               {"color": "#FF6F00", "marker": "D", "ls": "-",  "lw": 2.8,
                            "label": "HEUR2 (ours)"},
    "NEW_HEU":             {"color": "#E91E63", "marker": "D", "ls": "--", "lw": 1.8,
                            "label": "NEW_HEU"},
    "NEW_HEU_PR":          {"color": "#F44336", "marker": "X", "ls": "--", "lw": 1.8,
                            "label": "NEW_HEU_PR"},
    "ML_GUIDED":           {"color": "#3F51B5", "marker": "P", "ls": "-",  "lw": 2.8,
                            "label": "ML_GUIDED"},
    # --- Baselines ---
    "Myopic":              {"color": "#4CAF50", "marker": "o", "ls": "--", "lw": 2.0,
                            "label": "Myopic"},
    "NaiveMyopic":         {"color": "#607D8B", "marker": "s", "ls": ":",  "lw": 2.0,
                            "label": "Naive Myopic"},
    "Gonzales":            {"color": "#795548", "marker": "^", "ls": "-.", "lw": 2.0,
                            "label": "Gonzales"},
}

from attempt2.algorithms import ALGO_ORDER  # ["MYOPIC_HYBRID", "Myopic", "NaiveMyopic", "Gonzales"]

K_COLORS   = ["#1976D2", "#F57C00", "#388E3C", "#D32F2F", "#8E24AA"]
H_COLORS   = {0.2: "#FFCDD2", 0.4: "#FFE082", 0.6: "#C8E6C9", 0.8: "#BBDEFB"}

BG_LIGHT   = "#FFFFFF"
AX_LIGHT   = "#F8F9FA"
GRID_COLOR = "#E9ECEF"
TEXT_COLOR = "#212529"
EDGE_COLOR = "#DEE2E6"

def _apply_light_style():
    plt.rcParams.update({
        "figure.facecolor":  BG_LIGHT,
        "axes.facecolor":    AX_LIGHT,
        "axes.edgecolor":    EDGE_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "legend.facecolor":  "#FFFFFF",
        "legend.edgecolor":  EDGE_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "legend.framealpha": 0.9,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "figure.dpi":        130,
    })

_apply_light_style()


# ── Naming helpers ───────────────────────────────────────────────────────────

def _safe(val):
    """Sanitise a numeric value for use in filenames: '0.3' → '03'."""
    return str(val).replace(".", "").replace("-", "")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    -> {os.path.basename(path)}")


def _alpha_tag(alpha):
    return f"alpha{_safe(alpha)}"


def _h_tag(h):
    return f"h{_safe(h)}"


def _dataset_tag(dataset_name):
    """Sanitise dataset name for filenames."""
    return dataset_name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _algo_present(k_results):
    return [a for a in ALGO_ORDER if k_results.get(a) is not None]


def _detect_dataset_name(h_key):
    """
    Infer dataset name from the h key used in results dict.

    - If h_key is a float (0.2, 0.4, ...) → HICHBA synthetic network
    - If h_key is a string (e.g., 'ego-Facebook') → that's the dataset name
    """
    if isinstance(h_key, (int, float)):
        return "HICHBA"
    return str(h_key)


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_A_time_vs_variance(results, alpha, h, k, T, out_dir, dataset_name="HICHBA"):
    k_results = results.get(alpha, {}).get(h, {}).get(k, {})
    algos = _algo_present(k_results)
    if not algos:
        return

    t_axis = list(range(T + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(AX_LIGHT)

    for algo in ALGO_ORDER:
        res = k_results.get(algo)
        if res is None:
            continue
        st  = ALGO_STYLE.get(algo)
        if st is None:
            continue
        var_t = res["ts_metrics"]["var"]
        ax.plot(t_axis, var_t,
                color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=st["lw"],
                markersize=7, label=st["label"])

    ax.set_xlabel("Diffusion time step  t", color=TEXT_COLOR)
    ax.set_ylabel("Variance  Var(P^(t))", color=TEXT_COLOR)
    ax.set_xticks(t_axis)
    ax.set_title(
        f"Time vs Variance  |  α={alpha}   {dataset_name} h={h}   k={k}\n"
        f"Lower variance = fairer information access",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    fname = (f"{_dataset_tag(dataset_name)}_variance_vs_timestep_k{k}.png")
    _save(fig, os.path.join(out_dir, fname))


def plot_B_time_vs_k_variance(results, alpha, h, k_vals, T, out_dir, dataset_name="HICHBA"):
    t_axis = list(range(T + 1))

    for algo in ALGO_ORDER:
        st = ALGO_STYLE.get(algo)
        if st is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_LIGHT)
        ax.set_facecolor(AX_LIGHT)

        has_data = False
        for ki, k in enumerate(sorted(k_vals)):
            res = results.get(alpha, {}).get(h, {}).get(k, {}).get(algo)
            if res is None:
                continue
            var_t = res["ts_metrics"]["var"]
            color = K_COLORS[ki % len(K_COLORS)]
            ax.plot(t_axis, var_t,
                    color=color, marker="o", markersize=5,
                    linewidth=2.0, label=f"k={k}")
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Diffusion time step  t", color=TEXT_COLOR)
        ax.set_ylabel("Variance  Var(P^(t))", color=TEXT_COLOR)
        ax.set_xticks(t_axis)
        ax.set_title(
            f"Time vs Variance by Seed Size  |  {st['label']}\n"
            f"α={alpha}   {dataset_name} h={h}   (each line = different k)",
            fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.legend(fontsize=9, title="Seed size k",
                  title_fontsize=8)
        fig.tight_layout()

        safe_algo = algo.replace("-", "").replace(" ", "")
        fname = f"{_dataset_tag(dataset_name)}_time_vs_k_variance_{safe_algo}.png"
        _save(fig, os.path.join(out_dir, fname))


def plot_C_variance_vs_k(results, alpha, h, k_vals, out_dir, dataset_name="HICHBA"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(AX_LIGHT)

    k_sorted = sorted(k_vals)
    has_data = False
    for algo in ALGO_ORDER:
        st = ALGO_STYLE.get(algo)
        if st is None:
            continue
        vals = []
        for k in k_sorted:
            res = results.get(alpha, {}).get(h, {}).get(k, {}).get(algo)
            vals.append(res["final"]["var"] if res else None)

        xs = [k for k, v in zip(k_sorted, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if not xs:
            continue

        ax.plot(xs, ys,
                color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=st["lw"],
                markersize=8, label=st["label"])
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Seed set size  k", color=TEXT_COLOR)
    ax.set_ylabel("Final Variance  Var(P^T)", color=TEXT_COLOR)
    ax.set_xticks(k_sorted)
    ax.set_title(
        f"Variance vs Seed Set Size  |  α={alpha}   {dataset_name} h={h}\n"
        f"Final variance at deadline T   (lower = fairer)",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    fname = f"{_dataset_tag(dataset_name)}_variance_vs_seedset.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_D_reach_vs_k(results, alpha, h, k_vals, out_dir, dataset_name="HICHBA"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(AX_LIGHT)

    k_sorted = sorted(k_vals)
    for algo in ALGO_ORDER:
        st = ALGO_STYLE.get(algo)
        if st is None:
            continue
        vals = []
        for k in k_sorted:
            res = results.get(alpha, {}).get(h, {}).get(k, {}).get(algo)
            vals.append(res["final"]["mu"] if res else None)
        xs = [k for k, v in zip(k_sorted, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=st["lw"],
                markersize=8, label=st["label"])

    ax.set_xlabel("Seed set size  k", color=TEXT_COLOR)
    ax.set_ylabel("Mean reach  μ^T", color=TEXT_COLOR)
    ax.set_xticks(k_sorted)
    ax.set_title(
        f"Reach vs Seed Set Size  |  α={alpha}   {dataset_name} h={h}\n"
        f"Verifying ≤5% reach loss vs Myopic",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    fname = f"{_dataset_tag(dataset_name)}_reach_vs_seedset.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_E_disparity_vs_k(results, alpha, h, k_vals, out_dir, dataset_name="HICHBA"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(AX_LIGHT)

    k_sorted = sorted(k_vals)
    for algo in ALGO_ORDER:
        st = ALGO_STYLE.get(algo)
        if st is None:
            continue
        vals = []
        for k in k_sorted:
            res = results.get(alpha, {}).get(h, {}).get(k, {}).get(algo)
            vals.append(res["final"].get("disparity") if res else None)
        xs = [k for k, v in zip(k_sorted, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=st["lw"],
                markersize=8, label=st["label"])

    ax.set_xlabel("Seed set size  k", color=TEXT_COLOR)
    ax.set_ylabel("TCIM Group Disparity", color=TEXT_COLOR)
    ax.set_xticks(k_sorted)
    ax.set_title(
        f"TCIM Disparity vs Seed Size  |  α={alpha}   {dataset_name} h={h}\n"
        f"Ali et al. (2023) Eq.(2) — lower = fairer across groups",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    fname = f"{_dataset_tag(dataset_name)}_disparity_vs_seedset.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_F_vs_homophily(results, alpha, k_vals, h_vals, out_dir, dataset_name="HICHBA"):
    k_mid = sorted(k_vals)[len(k_vals) // 2]
    h_sorted = sorted(h_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    for ax in (ax1, ax2):
        ax.set_facecolor(AX_LIGHT)

    fig.suptitle(
        f"Final Metrics vs Homophily h  |  α={alpha}   {dataset_name}   k={k_mid}\n"
        f"Variance (left) and TCIM Disparity (right)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR)

    for algo in ALGO_ORDER:
        st = ALGO_STYLE.get(algo)
        if st is None:
            continue
        vars_ = []
        disps = []
        for h in h_sorted:
            res = results.get(alpha, {}).get(h, {}).get(k_mid, {}).get(algo)
            if res:
                vars_.append(res["final"]["var"])
                disps.append(res["final"].get("disparity", 0.0))
            else:
                vars_.append(None)
                disps.append(None)

        xs = [h for h, v in zip(h_sorted, vars_) if v is not None]
        ys_var  = [v for v in vars_ if v is not None]
        ys_disp = [v for v in disps if v is not None]

        kw = dict(color=st["color"], marker=st["marker"],
                  linestyle=st["ls"], linewidth=st["lw"],
                  markersize=8, label=st["label"])
        if xs:
            ax1.plot(xs, ys_var,  **kw)
            ax2.plot(xs, ys_disp, **kw)

    for ax, ylabel, note in [
        (ax1, "Variance  Var(P^T)",     "lower = fairer"),
        (ax2, "TCIM Group Disparity",   "lower = fairer across groups"),
    ]:
        ax.set_xlabel("Homophily  h", color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_xticks(h_sorted)
        ax.set_title(note, fontsize=9)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fname = f"{_dataset_tag(dataset_name)}_variance_vs_homophily_k{k_mid}.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_G_dashboard(results, alpha, h_vals, k_vals, T, out_dir, dataset_name="HICHBA"):
    h_sorted = sorted(h_vals)[:4]
    k_sorted = sorted(k_vals)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_LIGHT)
    for ax in axes.flat:
        ax.set_facecolor(AX_LIGHT)

    fig.suptitle(
        f"Variance vs Seed Size — All Homophily Levels  |  α={alpha}   {dataset_name}\n"
        f"Deadline T={T}  |  Lower variance = fairer access",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01)

    for ax, h in zip(axes.flat, h_sorted):
        for algo in ALGO_ORDER:
            st = ALGO_STYLE.get(algo)
            if st is None:
                continue
            vals = []
            for k in k_sorted:
                res = results.get(alpha, {}).get(h, {}).get(k, {}).get(algo)
                vals.append(res["final"]["var"] if res else None)
            xs = [k for k, v in zip(k_sorted, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if xs:
                ax.plot(xs, ys,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["ls"], linewidth=st["lw"],
                        markersize=7, label=st["label"])

        ax.set_title(f"h = {h}", fontsize=11, fontweight="bold",
                     color=H_COLORS.get(h, TEXT_COLOR))
        ax.set_xlabel("Seed set size  k")
        ax.set_ylabel("Variance  Var(P^T)")
        ax.set_xticks(k_sorted)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fname = f"{_dataset_tag(dataset_name)}_dashboard_variance.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_H_mivt_heatmap(results, alpha, h_vals, k_vals, out_dir, dataset_name="HICHBA"):
    """
    Heatmap: best 'ours' algorithm variance reduction vs Myopic.
    Picks whichever of our algorithms (PRISM*, DEG_DEFICIT*, FGD*) has lowest var.
    """
    h_sorted = sorted(h_vals)
    k_sorted = sorted(k_vals)

    # Our algorithms = everything except Myopic, NaiveMyopic, Gonzales
    baseline_algos = {"Myopic", "NaiveMyopic", "Gonzales"}
    ours_algos = [a for a in ALGO_ORDER if a not in baseline_algos]

    data = np.full((len(h_sorted), len(k_sorted)), np.nan)
    for hi, h in enumerate(h_sorted):
        for ki, k in enumerate(k_sorted):
            # Find best "ours" variance
            best_var_ours = float("inf")
            for algo in ours_algos:
                res = results.get(alpha,{}).get(h,{}).get(k,{}).get(algo)
                if res and res["final"]["var"] < best_var_ours:
                    best_var_ours = res["final"]["var"]

            res_myopic = results.get(alpha,{}).get(h,{}).get(k,{}).get("Myopic")
            if res_myopic and best_var_ours < float("inf"):
                var_myopic = res_myopic["final"]["var"]
                if var_myopic > 1e-9:
                    pct = (var_myopic - best_var_ours) / var_myopic * 100
                    data[hi, ki] = pct

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(AX_LIGHT)

    lim = max(abs(np.nanmin(data)) if not np.all(np.isnan(data)) else 1.0,
              abs(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0,
              1.0)
    im  = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                    vmin=-lim, vmax=lim)

    ax.set_xticks(range(len(k_sorted)))
    ax.set_xticklabels([f"k={k}" for k in k_sorted], fontsize=9)
    ax.set_yticks(range(len(h_sorted)))
    ax.set_yticklabels([f"h={h}" for h in h_sorted], fontsize=9)
    ax.set_xlabel("Seed set size  k")
    ax.set_ylabel("Homophily  h")

    for hi in range(len(h_sorted)):
        for ki in range(len(k_sorted)):
            v = data[hi, ki]
            if not np.isnan(v):
                ax.text(ki, hi, f"{v:+.1f}%",
                        ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("% variance reduction (Best Ours vs Myopic)",
                   color=TEXT_COLOR, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    fig.suptitle(
        f"Best-Ours Variance Reduction vs Myopic  |  α={alpha}   {dataset_name}\n"
        f"Green = Ours achieves lower variance (fairer)   "
        f"Red = Myopic wins",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout()

    fname = f"{_dataset_tag(dataset_name)}_heatmap_reduction.png"
    _save(fig, os.path.join(out_dir, fname))


def plot_I_group_timeseries(results, alpha, h, k, T, out_dir, dataset_name="HICHBA"):
    k_results = results.get(alpha, {}).get(h, {}).get(k, {})
    algos = _algo_present(k_results)
    if len(algos) == 0:
        return

    t_axis = list(range(T + 1))
    GROUP_COLORS = ["#1976D2", "#F57C00", "#388E3C", "#D32F2F", "#8E24AA"]

    n_algos = len(algos)
    # Cap columns to avoid overly wide figures
    n_cols = min(n_algos, 6)
    n_rows = (n_algos + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 5 * n_rows),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(BG_LIGHT)

    fig.suptitle(
        f"Per-Group Mean Reach over Time  |  α={alpha}   {dataset_name} h={h}   k={k}\n"
        f"Each line = one demographic group",
        fontsize=11, fontweight="bold", color=TEXT_COLOR)

    for idx, algo in enumerate(algos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor(AX_LIGHT)
        res = k_results.get(algo)
        if res is None:
            ax.set_title(f"{algo}\n(failed)", color=TEXT_COLOR)
            continue

        gts = res.get("group_reach_t", {})
        for gi, (g, vals) in enumerate(sorted(gts.items())):
            ax.plot(t_axis, vals,
                    color=GROUP_COLORS[gi % len(GROUP_COLORS)],
                    marker="o", markersize=4, linewidth=1.8,
                    label=f"Group {g}")

        st = ALGO_STYLE.get(algo, {"label": algo, "color": TEXT_COLOR})
        ax.set_title(st["label"], fontsize=10, color=st.get("color", TEXT_COLOR),
                     fontweight="bold")
        ax.set_xlabel("Diffusion time step  t")
        ax.set_xticks(t_axis)
        if col == 0:
            ax.set_ylabel("Mean p_v^(t) per group")
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused subplots
    for idx in range(len(algos), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fname = f"{_dataset_tag(dataset_name)}_group_reach_vs_timestep_k{k}.png"
    _save(fig, os.path.join(out_dir, fname))


# ── Master generation ────────────────────────────────────────────────────────

def generate_all_plots(results, args, out_dir):
    """
    Generate all plots, organised as:
        {out_dir}/{network_name}/{alpha}/
            {network_name}_*.png

    For social-network runs:  h keys are strings (network names).
    For HICHBA synthetic runs: h keys are floats; network name = 'HICHBA'.
    """
    alpha_vals = args.alpha
    k_vals     = args.k
    T          = args.T

    total = 0

    for alpha in alpha_vals:
        h_keys = sorted(results.get(alpha, {}).keys())
        if not h_keys:
            continue

        # Decide whether h keys are network names (social) or homophily values (HICHBA)
        social_run = isinstance(h_keys[0], str)

        if social_run:
            # ── Social-network mode ─────────────────────────────────────────
            # Structure: {out_dir}/{network_name}/{alpha}/
            for h in h_keys:
                ds_name  = h                                       # e.g. "email-EuAll"
                net_dir  = os.path.join(out_dir, ds_name, f"icm_alpha_{alpha}")
                os.makedirs(net_dir, exist_ok=True)
                print(f"\n  [{ds_name}]  alpha={alpha}  -> {net_dir}")

                for k in k_vals:
                    plot_A_time_vs_variance(results, alpha, h, k, T, net_dir,
                                            dataset_name=ds_name)
                    total += 1

                plot_B_time_vs_k_variance(results, alpha, h, k_vals, T, net_dir,
                                           dataset_name=ds_name)
                total += len(ALGO_ORDER)

                plot_C_variance_vs_k(results, alpha, h, k_vals, net_dir,
                                      dataset_name=ds_name)
                total += 1

                plot_D_reach_vs_k(results, alpha, h, k_vals, net_dir,
                                   dataset_name=ds_name)
                total += 1

                plot_E_disparity_vs_k(results, alpha, h, k_vals, net_dir,
                                       dataset_name=ds_name)
                total += 1

                k_rep = sorted(k_vals)[len(k_vals) // 2]
                plot_I_group_timeseries(results, alpha, h, k_rep, T, net_dir,
                                        dataset_name=ds_name)
                total += 1

        else:
            # ── HICHBA synthetic mode ───────────────────────────────────────
            # Structure: {out_dir}/HICHBA/{alpha}/
            dataset_name = "HICHBA"
            net_dir = os.path.join(out_dir, dataset_name, f"icm_alpha_{alpha}")
            os.makedirs(net_dir, exist_ok=True)
            print(f"\n  [HICHBA]  alpha={alpha}  -> {net_dir}")

            for h in h_keys:
                print(f"    h={h}:")
                for k in k_vals:
                    plot_A_time_vs_variance(results, alpha, h, k, T, net_dir,
                                            dataset_name=dataset_name)
                    total += 1

                plot_B_time_vs_k_variance(results, alpha, h, k_vals, T, net_dir,
                                           dataset_name=dataset_name)
                total += len(ALGO_ORDER)

                plot_C_variance_vs_k(results, alpha, h, k_vals, net_dir,
                                      dataset_name=dataset_name)
                total += 1

                plot_D_reach_vs_k(results, alpha, h, k_vals, net_dir,
                                   dataset_name=dataset_name)
                total += 1

                plot_E_disparity_vs_k(results, alpha, h, k_vals, net_dir,
                                       dataset_name=dataset_name)
                total += 1

                k_rep = sorted(k_vals)[len(k_vals) // 2]
                plot_I_group_timeseries(results, alpha, h, k_rep, T, net_dir,
                                        dataset_name=dataset_name)
                total += 1

            # Cross-homophily summary plots (go in the same net_dir)
            plot_F_vs_homophily(results, alpha, k_vals, h_keys, net_dir,
                                dataset_name=dataset_name)
            total += 1

            plot_G_dashboard(results, alpha, h_keys, k_vals, T, net_dir,
                              dataset_name=dataset_name)
            total += 1

            plot_H_mivt_heatmap(results, alpha, h_keys, k_vals, net_dir,
                                 dataset_name=dataset_name)
            total += 1

    print(f"\n  Total plots generated: {total}")



if __name__ == "__main__":
    import argparse
    import json
    
    p = argparse.ArgumentParser(description="Generate plots from an existing results.json file")
    p.add_argument("--results", type=str, default=None, help="Path to results.json")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for plots")
    cmd_args = p.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Auto-detect results file and network directory if not provided
    if cmd_args.results is None:
        if os.path.exists(os.path.join(root_dir, "results_network", "results.json")):
            cmd_args.results = os.path.join(root_dir, "results_network", "results.json")
            network_dir = os.path.join(root_dir, "Social_network")
        elif os.path.exists(os.path.join(root_dir, "results", "results.json")):
            cmd_args.results = os.path.join(root_dir, "results", "results.json")
            network_dir = os.path.join(root_dir, "Synthetic Networks")
        elif os.path.exists(os.path.join(root_dir, "attempt2", "results", "results.json")):
            cmd_args.results = os.path.join(root_dir, "attempt2", "results", "results.json")
            network_dir = os.path.join(root_dir, "Synthetic Networks")
        else:
            print("ERROR: Could not automatically find results.json in 'results_network' or 'results' folders.")
            sys.exit(1)
    else:
        # Best guess for network directory if manually provided
        if "results_network" in cmd_args.results:
            network_dir = os.path.join(root_dir, "Social_network")
        else:
            network_dir = os.path.join(root_dir, "Synthetic Networks")
            
    if cmd_args.out_dir is None:
        cmd_args.out_dir = os.path.join(os.path.dirname(cmd_args.results), "plots")
    
    print(f"Loading results from {cmd_args.results}...")
    with open(cmd_args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    class DummyArgs:
        pass
        
    args = DummyArgs()
    params = data.get("params", {})
    args.alpha = params.get("alpha", [0.075, 0.15, 0.3])
    args.k = params.get("k", [20, 40, 60, 80, 100])
    args.T = params.get("T", 8)
    
    results_data = data.get("results", {})
    
    parsed_results = {}
    for alpha_str, h_dict in results_data.items():
        try:
            a_key = float(alpha_str)
        except:
            a_key = alpha_str
            
        parsed_results[a_key] = {}
        for h_str, k_dict in h_dict.items():
            try:
                h_key = float(h_str)
            except:
                h_key = h_str
                
            parsed_results[a_key][h_key] = {}
            for k_str, algo_dict in k_dict.items():
                parsed_results[a_key][h_key][int(k_str)] = algo_dict
                
    os.makedirs(cmd_args.out_dir, exist_ok=True)
    generate_all_plots(parsed_results, args, cmd_args.out_dir)

    # # Seed node analysis — commented out
    # import subprocess
    # print(f"\nGenerating seed node analysis...")
    # analysis_script = os.path.join(root_dir, "analyze_result_seeds.py")
    # out_md = os.path.join(cmd_args.out_dir, "seed_analysis.md")
    # cmd = [
    #     sys.executable, analysis_script,
    #     "--results", cmd_args.results,
    #     "--network_dir", network_dir,
    #     "--out_file", out_md
    # ]
    # try:
    #     subprocess.run(cmd, check=True)
    # except Exception as e:
    #     print(f"Failed to run seed analysis: {e}")
