import os
import json
import argparse

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def parse_prop_algo_name(algo_name):
    """
    Extracts base family name, L, and lam from an algorithm name string like
    'Prop-Aware Additive (L=1, lam=0.1)'
    """
    if " (L=" not in algo_name:
        return None, None, None
    base_name = algo_name.split(" (")[0]
    param_part = algo_name.split(" (")[1].replace(")", "")
    parts = param_part.split(", ")
    l_val = int(parts[0].split("=")[1])
    lam_val = float(parts[1].split("=")[1])
    return base_name, l_val, lam_val

def render_svg_symbol(shape_type, cx, cy, color):
    """Renders distinct SVG markers for propagation-aware algorithm curves."""
    if shape_type == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>'
    elif shape_type == "square":
        return f'<rect x="{cx-4:.1f}" y="{cy-4:.1f}" width="8" height="8" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>'
    elif shape_type == "triangle":
        p1 = f"{cx:.1f},{cy-5.5:.1f}"
        p2 = f"{cx-5.0:.1f},{cy+4.5:.1f}"
        p3 = f"{cx+5.0:.1f},{cy+4.5:.1f}"
        return f'<polygon points="{p1} {p2} {p3}" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>'
    elif shape_type == "diamond":
        p1 = f"{cx:.1f},{cy-5.5:.1f}"
        p2 = f"{cx-5.5:.1f},{cy:.1f}"
        p3 = f"{cx:.1f},{cy+5.5:.1f}"
        p4 = f"{cx+5.5:.1f},{cy:.1f}"
        return f'<polygon points="{p1} {p2} {p3} {p4}" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>'
    else:
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{color}"/>'

def generate_svg_chart(network, alpha_str, k_list, prop_selections, baseline_data, prop_families, baselines, metric_name, output_svg_path, color_map):
    """
    Generates a standalone SVG chart comparing all 18 algorithm curves, with:
    - Point annotations showing the exact (L, lambda) combination chosen for each k.
    - Parameter breakdown in the side legend.
    """
    width = 1250
    height = 720
    margin_left = 90
    margin_right = 430
    margin_top = 70
    margin_bottom = 80

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_y = []
    for fam in prop_families:
        for k_int in k_list:
            info = prop_selections[fam].get(k_int)
            if info and info['val'] is not None:
                all_y.append(info['val'])

    for b_name in baselines:
        for k_int in k_list:
            val = baseline_data.get(b_name, {}).get(k_int)
            if val is not None:
                all_y.append(val)

    if not all_y:
        return

    min_y = min(all_y)
    max_y = max(all_y)

    if max_y == min_y:
        max_y += 0.01
        min_y = max(0, min_y - 0.01)
    else:
        y_range = max_y - min_y
        min_y = max(0, min_y - 0.05 * y_range)
        max_y = max_y + 0.05 * y_range

    min_x = min(k_list)
    max_x = max(k_list)
    x_range = max(1, max_x - min_x)

    def get_x_pixel(k_val):
        return margin_left + ((k_val - min_x) / x_range) * plot_w

    def get_y_pixel(y_val):
        return margin_top + plot_h - ((y_val - min_y) / (max_y - min_y)) * plot_h

    prop_symbols = {
        "Prop-Aware Additive": "circle",
        "Prop-Aware Multiplicative": "square",
        "Prop-Aware Additive KCore": "triangle",
        "Prop-Aware Multiplicative KCore": "diamond"
    }

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">')
    svg.append('<style>')
    svg.append('  text { font-family: system-ui, -apple-system, sans-serif; }')
    svg.append('  .title { font-size: 16px; font-weight: bold; fill: #111827; }')
    svg.append('  .axis-label { font-size: 13px; font-weight: 600; fill: #374151; }')
    svg.append('  .tick-text { font-size: 11px; fill: #4b5563; }')
    svg.append('  .grid { stroke: #e5e7eb; stroke-width: 1; stroke-dasharray: 4,4; }')
    svg.append('  .axis { stroke: #9ca3af; stroke-width: 1.5; }')
    svg.append('  .legend-text { font-size: 11px; fill: #1f2937; }')
    svg.append('  .param-tag { font-size: 9px; font-weight: 600; fill: #111827; }')
    svg.append('</style>')

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    # Title
    if metric_name == "variance":
        metric_title = "Variance of Access Probabilities"
    elif metric_name == "min_p":
        metric_title = "Rawlsian Fairness (Min Access Probability)"
    else:
        metric_title = "Access Probability (Spreading Reach)"

    svg.append(f'<text x="{margin_left + plot_w/2}" y="35" text-anchor="middle" class="title">{metric_title} vs. Seed Set Size (k) - {network} (alpha={alpha_str})</text>')

    # Grid & Y-Ticks
    y_ticks_count = 6
    for i in range(y_ticks_count + 1):
        y_val = min_y + (i / y_ticks_count) * (max_y - min_y)
        y_px = get_y_pixel(y_val)
        svg.append(f'<line x1="{margin_left}" y1="{y_px}" x2="{margin_left + plot_w}" y2="{y_px}" class="grid"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y_px + 4}" text-anchor="end" class="tick-text">{y_val:.4f}</text>')

    # X-Ticks
    for k_val in k_list:
        x_px = get_x_pixel(k_val)
        svg.append(f'<line x1="{x_px}" y1="{margin_top}" x2="{x_px}" y2="{margin_top + plot_h}" class="grid"/>')
        svg.append(f'<text x="{x_px}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="tick-text">{k_val}</text>')

    # Axes
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')

    # Axis Labels
    svg.append(f'<text x="{margin_left + plot_w/2}" y="{margin_top + plot_h + 50}" text-anchor="middle" class="axis-label">Seed Set Size (k)</text>')
    svg.append(f'<text x="25" y="{margin_top + plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {margin_top + plot_h/2})" class="axis-label">{metric_title}</text>')

    # 1. Draw Baseline Curves (SOLID Lines, Uniform stroke-width=2.0)
    for b_name in baselines:
        points = []
        for k_int in k_list:
            val = baseline_data.get(b_name, {}).get(k_int)
            if val is not None:
                points.append((get_x_pixel(k_int), get_y_pixel(val)))

        if points:
            color = color_map[b_name]
            path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.0"/>')

            for px, py in points:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="{color}"/>')

    # 2. Draw 4 Propagation-Aware Minimum Envelopes (DOTTED Lines stroke-dasharray="3,3", stroke-width=2.0, Distinct Symbols + (L, λ) annotations)
    for fam_idx, fam in enumerate(prop_families):
        points = []
        param_labels = []
        for k_int in k_list:
            info = prop_selections[fam].get(k_int)
            if info and info['val'] is not None:
                px = get_x_pixel(k_int)
                py = get_y_pixel(info['val'])
                points.append((px, py))
                param_labels.append((px, py, f"L{info['L']},λ{info['lam']:.1f}"))

        if points:
            color = color_map[fam]
            path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.0" stroke-dasharray="3,3"/>')

            symbol_type = prop_symbols[fam]
            for idx, (px, py) in enumerate(points):
                svg.append(render_svg_symbol(symbol_type, px, py, color))

                # Render small text tag indicating (L, lambda) for this point
                # Offset y slightly per family to prevent text overlap
                y_offset = -10 - (fam_idx * 2) if fam_idx % 2 == 0 else 14 + (fam_idx * 2)
                tag_str = param_labels[idx][2]
                svg.append(f'<text x="{px:.1f}" y="{py + y_offset:.1f}" text-anchor="middle" class="param-tag">{tag_str}</text>')

    # Legend Rendering (Right Margin)
    leg_x = margin_left + plot_w + 20
    leg_y = margin_top + 5

    svg.append(f'<rect x="{leg_x}" y="{leg_y - 10}" width="390" height="590" fill="#f9fafb" stroke="#e5e7eb" rx="6"/>')
    svg.append(f'<text x="{leg_x + 10}" y="{leg_y + 10}" font-weight="bold" class="legend-text">Propagation-Aware Envelopes (Dotted + Symbols):</text>')

    leg_curr_y = leg_y + 28
    for fam in prop_families:
        color = color_map[fam]
        symbol_type = prop_symbols[fam]
        svg.append(f'<line x1="{leg_x + 10}" y1="{leg_curr_y}" x2="{leg_x + 35}" y2="{leg_curr_y}" stroke="{color}" stroke-width="2.0" stroke-dasharray="3,3"/>')
        svg.append(render_svg_symbol(symbol_type, leg_x + 22.5, leg_curr_y, color))
        svg.append(f'<text x="{leg_x + 42}" y="{leg_curr_y + 4}" class="legend-text" font-weight="600">{fam}</text>')

        # Add brief summary of (L, lambda) choices across k
        param_seq = []
        for k_int in k_list[:5]: # show first 5 k values summary
            info = prop_selections[fam].get(k_int)
            if info:
                param_seq.append(f"k{k_int}:L{info['L']},λ{info['lam']:.1f}")
        summary_str = " ".join(param_seq)
        svg.append(f'<text x="{leg_x + 42}" y="{leg_curr_y + 16}" font-size="8.5px" fill="#4b5563">{summary_str}</text>')

        leg_curr_y += 32

    leg_curr_y += 5
    svg.append(f'<text x="{leg_x + 10}" y="{leg_curr_y}" font-weight="bold" class="legend-text">Baseline Algorithms (Solid Lines):</text>')
    leg_curr_y += 20

    for b_name in baselines:
        color = color_map[b_name]
        svg.append(f'<line x1="{leg_x + 10}" y1="{leg_curr_y}" x2="{leg_x + 35}" y2="{leg_curr_y}" stroke="{color}" stroke-width="2.0"/>')
        svg.append(f'<circle cx="{leg_x + 22.5}" cy="{leg_curr_y}" r="3" fill="{color}"/>')
        svg.append(f'<text x="{leg_x + 42}" y="{leg_curr_y + 4}" class="legend-text">{b_name}</text>')
        leg_curr_y += 18

    svg.append('</svg>')

    with open(output_svg_path, "w") as f:
        f.write("\n".join(svg))

def main():
    parser = argparse.ArgumentParser(description="Plot access probabilities / variance vs seedset with explicit (L, lambda) annotations.")
    parser.add_argument("--input", type=str, default=None, help="Path to master.json file (e.g. var_1/master.json or acc/master.json)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to parent folder of --input)")
    parser.add_argument("--metric", type=str, default=None, choices=["variance", "min_p", "mean_prob"], help="Metric to optimize and plot (default: variance for var_1, min_p for acc)")
    args = parser.parse_args()

    if args.input is None:
        if os.path.exists("var_1/master.json"):
            args.input = "var_1/master.json"
        elif os.path.exists("acc/master.json"):
            args.input = "acc/master.json"
        else:
            print("Error: Please specify --input path to master.json")
            return

    if not os.path.exists(args.input):
        print(f"Error: Input JSON file not found at {args.input}")
        return

    if "acc" in args.input and args.metric is None:
        args.metric = "min_p"
    elif args.metric is None:
        args.metric = "variance"

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))

    print(f"Loading experiment data from {args.input}...")
    print(f"Target metric: '{args.metric}' | Output directory: '{args.output_dir}'")

    with open(args.input, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    baselines = [
        "Random", "Myopic", "Naive Myopic", "Gonzalez",
        "Myopic BFS", "Naive Myopic BFS", "Myopic PPR", "Naive Myopic PPR",
        "LeastCentral", "LeastCentral_n",
        "MinDegree_hc", "MinDegree_hcn", "MinDegree_nd", "MinDegree_ndn"
    ]

    prop_families = [
        "Prop-Aware Additive",
        "Prop-Aware Multiplicative",
        "Prop-Aware Additive KCore",
        "Prop-Aware Multiplicative KCore"
    ]

    distinct_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#008080", "#b8860b",
        "#4b0082", "#ff1493", "#2e8b57", "#d2691e", "#4169e1", "#dc143c"
    ]

    color_map = {}
    all_names = prop_families + baselines
    for idx, name in enumerate(all_names):
        color_map[name] = distinct_colors[idx % len(distinct_colors)]

    mpl_prop_markers = {
        "Prop-Aware Additive": "o",
        "Prop-Aware Multiplicative": "s",
        "Prop-Aware Additive KCore": "^",
        "Prop-Aware Multiplicative KCore": "D"
    }

    for network, alphas in data.items():
        print(f"Processing network: {network}")
        for alpha_str, ks in alphas.items():
            try:
                alpha_val = float(alpha_str)
                alpha_fmt = f"alpha_{alpha_val:.2f}"
            except ValueError:
                alpha_fmt = f"alpha_{alpha_str}"

            folder_path = os.path.join(args.output_dir, network, alpha_fmt)
            os.makedirs(folder_path, exist_ok=True)

            k_list = sorted([int(k) for k in ks.keys()])

            prop_selections = {fam: {} for fam in prop_families}
            baseline_data = {b_name: {} for b_name in baselines}

            for k_int in k_list:
                k_str = str(k_int)
                algos_dict = ks[k_str]

                for b_name in baselines:
                    if b_name in algos_dict:
                        val = algos_dict[b_name].get(args.metric, None)
                        if val is not None:
                            baseline_data[b_name][k_int] = val

                for fam in prop_families:
                    best_val = None
                    best_info = None

                    for algo_name, metrics in algos_dict.items():
                        if algo_name.startswith(fam + " ("):
                            val = metrics.get(args.metric, None)
                            if val is None:
                                continue

                            base_name, l_val, lam_val = parse_prop_algo_name(algo_name)

                            if args.metric == "variance":
                                is_better = (best_val is None) or (val < best_val)
                            else:
                                is_better = (best_val is None) or (val > best_val)

                            if is_better:
                                best_val = val
                                best_info = {
                                    'L': l_val,
                                    'lam': lam_val,
                                    'val': val,
                                    'algo_name': algo_name
                                }

                    if best_info is not None:
                        prop_selections[fam][k_int] = best_info

            # Write parameter selection text file
            txt_report_path = os.path.join(folder_path, "selected_params_per_k.txt")
            with open(txt_report_path, "w") as txt_file:
                txt_file.write(f"=== Optimal Parameter Selection per Seed Set Size (k) ===\n")
                txt_file.write(f"Network: {network}\n")
                txt_file.write(f"Alpha: {alpha_str}\n")
                txt_file.write(f"Optimization Metric: {args.metric} ({'Minimizing' if args.metric == 'variance' else 'Maximizing'})\n")
                txt_file.write("="*70 + "\n\n")

                for k_int in k_list:
                    txt_file.write(f"--- Seed Set Size k = {k_int} ---\n")
                    for fam in prop_families:
                        info = prop_selections[fam].get(k_int, None)
                        if info:
                            txt_file.write(
                                f"  [{fam}]\n"
                                f"    Selected Parameters : L = {info['L']}, Weight (lambda) = {info['lam']:.1f}\n"
                                f"    Full Variant Name   : {info['algo_name']}\n"
                                f"    Resulting {args.metric:10s}: {info['val']:.6f}\n"
                            )
                        else:
                            txt_file.write(f"  [{fam}]\n    Selected Parameters : N/A\n")
                    txt_file.write("\n")

            # Chart filename
            file_prefix = "access_prob" if args.metric != "variance" else "variance"
            svg_chart_path = os.path.join(folder_path, f"{file_prefix}_comparison_all_algos.svg")
            generate_svg_chart(network, alpha_str, k_list, prop_selections, baseline_data, prop_families, baselines, args.metric, svg_chart_path, color_map)

            # Generate PNG Chart if matplotlib is available
            if HAS_MATPLOTLIB:
                png_chart_path = os.path.join(folder_path, f"{file_prefix}_comparison_all_algos.png")
                plt.figure(figsize=(14, 8))

                for fam in prop_families:
                    x_vals = []
                    y_vals = []
                    labels_list = []
                    for k_int in k_list:
                        info = prop_selections[fam].get(k_int)
                        if info:
                            x_vals.append(k_int)
                            y_vals.append(info['val'])
                            labels_list.append(f"L{info['L']},λ{info['lam']:.1f}")

                    if x_vals:
                        line = plt.plot(
                            x_vals, y_vals,
                            label=f"{fam} (Best {args.metric}/k)",
                            color=color_map[fam],
                            marker=mpl_prop_markers[fam],
                            linewidth=2.0,
                            markersize=6,
                            linestyle=":"
                        )
                        # Annotate points on PNG chart
                        for xi, yi, lbl in zip(x_vals, y_vals, labels_list):
                            plt.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=7, fontweight='bold')

                for idx, b_name in enumerate(baselines):
                    x_vals = []
                    y_vals = []
                    for k_int in k_list:
                        val = baseline_data.get(b_name, {}).get(k_int)
                        if val is not None:
                            x_vals.append(k_int)
                            y_vals.append(val)

                    if x_vals:
                        plt.plot(
                            x_vals, y_vals,
                            label=b_name,
                            color=color_map[b_name],
                            marker="o",
                            linewidth=2.0,
                            markersize=4,
                            linestyle="-",
                            alpha=0.85
                        )

                metric_label = "Variance of Access Probabilities" if args.metric == "variance" else "Rawlsian Fairness (Min Access Probability)"
                plt.title(f"{metric_label} vs. Seed Set Size (k) - {network} (alpha={alpha_str})", fontsize=12, fontweight='bold')
                plt.xlabel("Seed Set Size (k)", fontsize=11)
                plt.ylabel(metric_label, fontsize=11)
                plt.xticks(k_list)
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=True)
                plt.tight_layout()
                plt.savefig(png_chart_path, dpi=150)
                plt.close()

            print(f"  Saved params : {txt_report_path}")
            print(f"  Saved SVG    : {svg_chart_path}")

    print(f"\nAll plots with explicit (L, lambda) annotations successfully saved in '{args.output_dir}'!")

if __name__ == "__main__":
    main()
