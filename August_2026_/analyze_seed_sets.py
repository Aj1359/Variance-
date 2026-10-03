import os
import glob
import json
import networkx as nx

def main():
    print("Starting Seed Set Analysis...")
    
    # 1. Find all result folders (excluding result_master)
    result_folders = sorted([d for d in glob.glob("result_*") if os.path.isdir(d) and "result_master" not in d])
    if not result_folders:
        print("No result folders found.")
        return
        
    gml_cache = {}
    
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/SEED_SET_ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Seed Set Analysis Report\n\n")
        f.write("This report analyzes the structural attributes (Degree, PageRank) of the seed nodes selected by each algorithm.\n\n")
        f.write("---\n\n")
        
        for folder in result_folders:
            algo_name = folder.replace("result_", "").replace("_multi", "").replace("_", " ").title()
            f.write(f"## Heuristic Algorithm: {algo_name}\n\n")
            
            # Find results.json files in this folder
            json_files = glob.glob(os.path.join(folder, "**", "results.json"), recursive=True)
            if not json_files:
                f.write("No results found for this heuristic.\n\n")
                continue
                
            for json_path in sorted(json_files):
                # Path parts: folder/network/alpha/results.json
                norm_path = json_path.replace("\\", "/")
                parts = norm_path.split("/")
                network = parts[1] if len(parts) > 1 else "Unknown"
                alpha = parts[2] if len(parts) > 2 else "Unknown"
                
                # Load GML to cache to translate node IDs and fetch attributes
                gml_key = network
                if gml_key not in gml_cache:
                    gml_path = os.path.join("Social_Network", f"{network}.gml")
                    if os.path.exists(gml_path):
                        print(f"Loading GML for {network}...")
                        G = nx.read_gml(gml_path, destringizer=int)
                        original_nodes = sorted([int(n) for n in G.nodes()])
                        gml_cache[gml_key] = {
                            "G": G,
                            "original_nodes": original_nodes
                        }
                    else:
                        gml_cache[gml_key] = None
                        
                gml_data = gml_cache[gml_key]
                if not gml_data:
                    continue
                    
                G = gml_data["G"]
                original_nodes = gml_data["original_nodes"]
                
                with open(json_path, "r") as jf:
                    try:
                        data = json.load(jf)
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")
                        continue
                
                # We want to display the target algorithm's seed nodes
                # Find target algorithm key in the json keys
                target_key = None
                for key in data.keys():
                    if key not in ["myopic", "naive_myopic", "gonzalez"]:
                        target_key = key
                        break
                        
                if not target_key:
                    continue
                    
                f.write(f"### Dataset: {network} | Alpha: {alpha}\n\n")
                
                target_data = data[target_key]
                # Target data is keyed by k (str), e.g. "5", "10", "15"
                sorted_ks = sorted([int(k) for k in target_data.keys()])
                
                for k in sorted_ks:
                    k_str = str(k)
                    seeds = target_data[k_str].get("seeds", [])
                    if not seeds:
                        continue
                        
                    f.write(f"#### Seed Set for k = {k}\n\n")
                    f.write("| Selection Order | Original Node ID | Mapped Node ID | Degree | PageRank |\n")
                    f.write("|---|---|---|---|---|\n")
                    
                    for order, mapped_id in enumerate(seeds, 1):
                        orig_id = original_nodes[mapped_id]
                        # Fetch attributes
                        degree = G.degree[orig_id]
                        node_attrs = G.nodes[orig_id]
                        pagerank = 0.0
                        for pr_key in ["pagerank_centrality", "pagerank", "page_rank", "PageRank"]:
                            if pr_key in node_attrs:
                                pagerank = float(node_attrs[pr_key])
                                break
                        f.write(f"| {order} | {orig_id} | {mapped_id} | {degree} | {pagerank:.6f} |\n")
                    f.write("\n")
                f.write("---\n\n")
                
    print(f"Seed set analysis written successfully to {report_path}!")

if __name__ == "__main__":
    main()
