import os
import glob
import random
import networkx as nx
import numpy as np
import time

def get_hub_count(G):
    """
    Returns the number of high-degree hubs.
    We define a hub as a node with a degree > 5 * average_degree.
    """
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return 0
    avg_deg = sum(degrees) / len(degrees)
    threshold = 5 * avg_deg
    hubs = [d for d in degrees if d > threshold]
    return len(hubs)

def estimate_path_and_diameter(G, num_samples=100):
    """
    Computes the average shortest path length and diameter for the current graph.

    We do not block on graph size: for smaller networks we use exact values, and for
    larger ones we still attempt a sampled estimate so the script keeps running for
    all network sizes without refusing to process them.
    """
    nodes = list(G.nodes())
    if not nodes:
        return 0.0, 0

    try:
        if len(nodes) <= 3000:
            return nx.average_shortest_path_length(G), nx.diameter(G)
    except Exception:
        pass

    print(f"      [!] Graph is large ({len(nodes)} nodes). Running sampled estimate so processing continues...")
    sampled_nodes = random.sample(nodes, min(len(nodes), num_samples))
    path_lengths = []
    max_diam = 0

    for source in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        distances = [d for n, d in lengths.items() if n != source]
        if distances:
            path_lengths.extend(distances)
            local_max = max(distances)
            if local_max > max_diam:
                max_diam = local_max

    if not path_lengths:
        return 0.0, 0

    return float(np.mean(path_lengths)), max_diam


def verify_required_metrics(report_path):
    """Ensure the required statistics are actually present in the markdown output."""
    required_tokens = [
        "Final Hubs",
        "Modularity",
        "Diameter",
    ]
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return False, "Report file was not created."

    missing = [token for token in required_tokens if token not in text]
    if missing:
        return False, f"Missing required metric columns: {', '.join(missing)}"
    return True, "All required metric columns were found."

def compute_modularity_safe(G):
    """
    Computes Modularity safely. Louvain community detection is fast, but exact Modularity
    can still be expensive. Greedy modularity is O(N^3) and will freeze on large graphs.
    """
    nodes = G.number_of_nodes()
    if nodes > 100000:
        return "N/A (Too Large)"
        
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        comms = louvain_communities(G)
        mod = modularity(G, comms)
        return f"{mod:.4f}"
    except ImportError:
        # Louvain is missing in older NetworkX versions. Fallback to greedy
        if nodes > 5000:
            return "N/A (Requires Louvain)"
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        comms = greedy_modularity_communities(G)
        mod = modularity(G, comms)
        return f"{mod:.4f}"
    except Exception as e:
        return "Error"

def main():
    input_dir = "Social_Network"
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
        
    out_file = "graph_statistics.md"
    
    with open(out_file, "w") as f:
        f.write("# Graph Statistics\n\n")
        f.write("| Network | Nodes | Edges | Density | Final Hubs (>5x Avg Deg) | Modularity | Avg Shortest Path (LCC) | Diameter (LCC) |\n")
        f.write("|---------|-------|-------|---------|--------------------------|------------|-------------------------|----------------|\n")

        for filepath in sorted(txt_files):
            name = os.path.basename(filepath).replace(".txt", "")
            print(f"\nLoading {name}...")

            try:
                t0 = time.time()
                G = nx.read_edgelist(filepath, comments='#', create_using=nx.Graph(), nodetype=int, data=False)
                print(f"  -> Loaded in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  Error loading {name}: {e}")
                continue

            nodes = G.number_of_nodes()
            edges = G.number_of_edges()
            density = nx.density(G)

            print("  -> Counting hubs...")
            hubs = get_hub_count(G)

            print("  -> Computing modularity...")
            mod_str = compute_modularity_safe(G)

            print(f"  -> Nodes: {nodes}, Edges: {edges}, Density: {density:.6e}, Hubs: {hubs}, Mod: {mod_str}")

            if not nx.is_connected(G):
                print("  -> Graph disconnected. Extracting Largest Connected Component...")
                largest_cc = max(nx.connected_components(G), key=len)
                G_lcc = G.subgraph(largest_cc)
            else:
                G_lcc = G

            print("  -> Computing Paths and Diameter...")
            avg_path, diameter = estimate_path_and_diameter(G_lcc, num_samples=100)

            if G_lcc.number_of_nodes() <= 3000:
                path_str = f"{avg_path:.3f}"
                diam_str = f"{diameter}"
            else:
                path_str = f"~{avg_path:.3f}"
                diam_str = f"~{diameter}"

            print(f"  -> Avg Path: {path_str}, Diameter: {diam_str}")
            f.write(f"| {name} | {nodes:,} | {edges:,} | {density:.6e} | {hubs:,} | {mod_str} | {path_str} | {diam_str} |\n")
            f.flush()

    ok, msg = verify_required_metrics(out_file)
    print(f"\nAll stats saved to {out_file}!")
    print(f"  -> Metric verification: {msg}")
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
