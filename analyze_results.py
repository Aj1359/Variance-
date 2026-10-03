import argparse
import json
import os
from collections import defaultdict


def deep_merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value


def load_json_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    i = 0
    objects = []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON at offset {i}: {exc}") from exc
        objects.append(obj)
        i += end

    if not objects:
        raise ValueError(f"No JSON objects found in {path}")

    merged = {}
    for obj in objects:
        if isinstance(obj, dict):
            deep_merge(merged, obj)
    return merged


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_numeric_key(key):
    try:
        float(key)
        return True
    except (TypeError, ValueError):
        return False


def get_var_from_result(result):
    if not isinstance(result, dict):
        return None
    final = result.get("final", {})
    if not isinstance(final, dict):
        return None
    var = final.get("var")
    return safe_float(var)


def normalize_results(raw_results):
    """Normalize different JSON layouts into: network -> alpha -> k -> algorithm -> variance."""
    norm = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for outer_key, outer_value in raw_results.items():
        if not isinstance(outer_value, dict):
            continue

        if is_numeric_key(outer_key):
            alpha = float(outer_key)
            for network, network_value in outer_value.items():
                if not isinstance(network_value, dict):
                    continue
                for k_key, k_value in network_value.items():
                    if not isinstance(k_value, dict):
                        continue
                    for algo, result in k_value.items():
                        var = get_var_from_result(result)
                        if var is None:
                            continue
                        norm[network][alpha][str(k_key)][str(algo)] = var
        else:
            network = str(outer_key)
            for inner_key, inner_value in outer_value.items():
                if not isinstance(inner_value, dict):
                    continue
                if is_numeric_key(inner_key):
                    alpha = float(inner_key)
                    for k_key, k_value in inner_value.items():
                        if not isinstance(k_value, dict):
                            continue
                        for algo, result in k_value.items():
                            var = get_var_from_result(result)
                            if var is None:
                                continue
                            norm[network][alpha][str(k_key)][str(algo)] = var
                else:
                    # Some files nest as network -> alpha -> k -> algo directly.
                    for k_key, k_value in inner_value.items():
                        if not isinstance(k_value, dict):
                            continue
                        for algo, result in k_value.items():
                            var = get_var_from_result(result)
                            if var is None:
                                continue
                            norm[network][alpha][str(k_key)][str(algo)] = var
    return norm


def rank_by_key(data, grouping_name):
    """Return ranking list sorted by mean variance ascending."""
    agg = defaultdict(list)

    for key, alpha_map in data.items():
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                for algo, var in algo_map.items():
                    agg[algo].append(var)

    ranking = []
    for algo, values in agg.items():
        ranking.append({
            "algo": algo,
            "mean_var": sum(values) / len(values),
            "count": len(values),
        })

    ranking.sort(key=lambda item: (item["mean_var"], item["algo"]))
    return ranking


def print_rankings(title, entries):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    for rank, item in enumerate(entries, start=1):
        print(f"{rank:>2}. {item['algo']:<22} mean variance = {item['mean_var']:.8f}  (n={item['count']})")


def build_combo_results(norm_data):
    combo_results = []
    for network, alpha_map in norm_data.items():
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                if not algo_map:
                    continue
                best_algo = min(algo_map.items(), key=lambda item: item[1])[0]
                combo_results.append({
                    "network": network,
                    "alpha": alpha,
                    "k": int(k),
                    "best_algo": best_algo,
                    "best_var": min(algo_map.values()),
                    "algo_vars": algo_map,
                })
    return combo_results


def analyze_master(json_path, target_algo="MYOPIC_HYBRID"):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    print(f"Loading {json_path}...")
    data = load_json_documents(json_path)

    raw_results = data.get("results", {})
    if not raw_results:
        print("No 'results' section found in the input file.")
        return

    norm_data = normalize_results(raw_results)
    if not norm_data:
        print("No valid algorithm result entries were found in the file.")
        return

    # Network-wise ranking on average variance across alphas and k values.
    network_rankings = []
    for network, alpha_map in sorted(norm_data.items()):
        values = []
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                for algo, var in algo_map.items():
                    values.append(var)
        if values:
            network_rankings.append({
                "network": network,
                "mean_var": sum(values) / len(values),
                "count": len(values),
            })
    network_rankings.sort(key=lambda item: (item["mean_var"], item["network"]))

    # Alpha-wise ranking on average variance across networks and k values.
    alpha_rankings = []
    alpha_groups = defaultdict(list)
    for network, alpha_map in norm_data.items():
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                for algo, var in algo_map.items():
                    alpha_groups[alpha].append(var)
    for alpha, values in sorted(alpha_groups.items(), key=lambda x: float(x[0])):
        alpha_rankings.append({
            "alpha": alpha,
            "mean_var": sum(values) / len(values),
            "count": len(values),
        })
    alpha_rankings.sort(key=lambda item: (item["mean_var"], item["alpha"]))

    # Seed-count wise ranking (k values) on average variance across networks and alphas.
    k_rankings = []
    k_groups = defaultdict(list)
    for network, alpha_map in norm_data.items():
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                for algo, var in algo_map.items():
                    k_groups[int(k)].append(var)
    for k, values in sorted(k_groups.items()):
        k_rankings.append({
            "k": k,
            "mean_var": sum(values) / len(values),
            "count": len(values),
        })
    k_rankings.sort(key=lambda item: (item["mean_var"], item["k"]))

    # Algorithm-wise ranking across all network/alpha/k combinations.
    overall_algo_rankings = []
    algo_groups = defaultdict(list)
    for network, alpha_map in norm_data.items():
        for alpha, k_map in alpha_map.items():
            for k, algo_map in k_map.items():
                for algo, var in algo_map.items():
                    algo_groups[algo].append(var)
    for algo, values in sorted(algo_groups.items()):
        overall_algo_rankings.append({
            "algo": algo,
            "mean_var": sum(values) / len(values),
            "count": len(values),
        })
    overall_algo_rankings.sort(key=lambda item: (item["mean_var"], item["algo"]))

    # Print the per-view rankings.
    print_rankings("NETWORK-WISE VARIANCE RANKING", [
        {"algo": item["network"], "mean_var": item["mean_var"], "count": item["count"]}
        for item in network_rankings
    ])
    print_rankings("ALPHA-WISE VARIANCE RANKING", [
        {"algo": f"alpha={item['alpha']}", "mean_var": item["mean_var"], "count": item["count"]}
        for item in alpha_rankings
    ])
    print_rankings("SEED-COUNT-WISE VARIANCE RANKING", [
        {"algo": f"k={item['k']}", "mean_var": item["mean_var"], "count": item["count"]}
        for item in k_rankings
    ])
    print_rankings("OVERALL ALGORITHM VARIANCE RANKING", overall_algo_rankings)

    # Final target-algorithm win analysis across the 75 combinations.
    combo_results = build_combo_results(norm_data)
    total_combinations = len(combo_results)
    wins = 0
    ties = 0
    losses = 0

    print(f"\n{'=' * 80}")
    print(f"FINAL WIN ANALYSIS FOR TARGET: {target_algo}")
    print(f"Total combinations checked: {total_combinations}")
    print(f"{'=' * 80}")

    for item in combo_results:
        best_algo = item["best_algo"]
        if best_algo == target_algo:
            wins += 1
        elif target_algo in item["algo_vars"] and item["algo_vars"].get(target_algo) == item["best_var"]:
            ties += 1
        else:
            losses += 1

    print(f"Target {target_algo} wins: {wins} / {total_combinations}")
    print(f"Ties: {ties}")
    print(f"Losses: {losses}")
    print(f"Win rate: {wins / total_combinations * 100:.2f}%")

    print(f"\nDETAIL BY NETWORK / ALPHA / K")
    print("-" * 80)
    for item in sorted(combo_results, key=lambda x: (x["network"], float(x["alpha"]), x["k"])):
        network = item["network"]
        alpha = item["alpha"]
        k = item["k"]
        sort_order = sorted(item["algo_vars"].items(), key=lambda x: (x[1], x[0]))
        winner = sort_order[0][0]
        target_var = item["algo_vars"].get(target_algo)
        marker = "*" if winner == target_algo else " "
        print(f"{marker} Network={network:<12} Alpha={alpha:<5} k={k:<3} best={winner:<20} var={sort_order[0][1]:.8f}  target={target_var if target_var is not None else 'n/a'}")

    print(f"\nAlgorithm ranking by average variance:")
    for rank, item in enumerate(overall_algo_rankings, start=1):
        print(f"{rank:>2}. {item['algo']:<22} -> {item['mean_var']:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a variance results JSON and rank algorithms by mean final variance.")
    parser.add_argument("file", nargs="?", default="results_social/results.json", help="Path to the results JSON file.")
    parser.add_argument("--target", default="MYOPIC_HYBRID", help="Target algorithm for final win count analysis.")
    args = parser.parse_args()

    analyze_master(args.file, args.target)
