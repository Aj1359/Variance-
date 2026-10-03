"""
build_gml.py
============
Converts raw edge-list .txt files from Social_Networks/ -> enriched GML files in Socials/.

Memory-optimized version designed to handle 30M+ edge graphs (e.g. LiveJournal)
without running out of memory (OOM / Killed).

Avoids building a NetworkX graph structure in memory, using instead flat arrays
and streaming the output file.

Each GML file will have:
  - Node attribute: degree (int)
  - Node attribute: pagerank (float)
  - Edge attribute: weight 1.0

Usage (from August_2026_ directory on SSH):
    python build_gml.py
    python build_gml.py --src /path/to/Social_Networks --dst /path/to/Socials
"""

import os
import sys
import argparse
import time
import array

# Defaults: sibling of the script in August_2026_
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(_HERE, "Social_Networks")
DEFAULT_DST = os.path.join(_HERE, "Socials")


def parse_and_map_edgelist(path):
    """
    Reads a SNAP-style edge-list file line-by-line.
    Dynamically maps arbitrary node IDs to contiguous 0-indexed IDs.
    Returns:
        flat_edges: array('i') of mapped edges [u0, v0, u1, v1, ...]
        degrees: array('i') where index is mapped ID and value is degree
        node_map: dict mapping original ID to mapped ID
    """
    node_map = {}
    degrees = array.array('i')
    flat_edges = array.array('i')

    print(f"  Streaming edge list from {os.path.basename(path)} ...", flush=True)
    t0 = time.time()
    count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
                if u == v:
                    continue  # skip self-loops
                
                # Map u
                if u not in node_map:
                    node_map[u] = len(node_map)
                    degrees.append(0)
                # Map v
                if v not in node_map:
                    node_map[v] = len(node_map)
                    degrees.append(0)
                
                mu = node_map[u]
                mv = node_map[v]
                
                flat_edges.append(mu)
                flat_edges.append(mv)
                degrees[mu] += 1
                degrees[mv] += 1
                
                count += 1
                if count % 10000000 == 0:
                    print(f"    Loaded {count:,} edges ...", flush=True)
            except ValueError:
                continue

    print(f"  Loaded {count:,} edges and {len(node_map):,} unique nodes in {time.time()-t0:.1f}s", flush=True)
    return flat_edges, degrees, node_map


def compute_pagerank(num_nodes, flat_edges, degrees, alpha=0.85, max_iter=100, tol=1e-4):
    """
    Memory-efficient PageRank calculation using power iteration on flat arrays.
    Avoids building sparse matrices or NetworkX graph objects.
    """
    N = num_nodes
    if N == 0:
        return []

    print(f"  Computing PageRank (alpha={alpha}, max_iter={max_iter}, tol={tol}) ...", flush=True)
    t0 = time.time()
    
    # Initial probability vector
    PR = array.array('d', [1.0 / N] * N)
    damping_const = (1.0 - alpha) / N

    for it in range(max_iter):
        next_PR = array.array('d', [damping_const] * N)
        
        # Precompute contribution for each node to avoid division inside edge loop
        contrib = array.array('d', [0.0] * N)
        for i in range(N):
            deg = degrees[i]
            if deg > 0:
                contrib[i] = PR[i] / deg
        
        # Power iteration step: iterate over flat edges list
        for idx in range(0, len(flat_edges), 2):
            u = flat_edges[idx]
            v = flat_edges[idx+1]
            next_PR[u] += alpha * contrib[v]
            next_PR[v] += alpha * contrib[u]
            
        # Compute L1 norm differences for convergence check
        err = 0.0
        for i in range(N):
            err += abs(next_PR[i] - PR[i])
            
        PR = next_PR
        if err < tol:
            print(f"    Converged in {it+1} iterations (error={err:.2e})", flush=True)
            break
    else:
        print(f"    Reached max_iter={max_iter} iterations without convergence", flush=True)

    print(f"  PageRank done in {time.time()-t0:.1f}s", flush=True)
    return PR


def stream_to_gml(gml_path, degrees, pr, flat_edges):
    """
    Streams node and edge data directly to a GML text file to keep memory footprint O(1).
    Only writes unique edges where source < target.
    """
    print(f"  Streaming nodes and edges to {gml_path} ...", flush=True)
    t0 = time.time()
    num_nodes = len(degrees)
    
    with open(gml_path, "w", encoding="utf-8") as f:
        f.write("graph [\n")
        
        # Write nodes
        for i in range(num_nodes):
            f.write("  node [\n")
            f.write(f"    id {i}\n")
            f.write(f"    degree {degrees[i]}\n")
            f.write(f"    pagerank {pr[i]:.10f}\n")
            f.write("  ]\n")
            
        # Write edges (avoid writing duplicates by checking u < v)
        for idx in range(0, len(flat_edges), 2):
            u = flat_edges[idx]
            v = flat_edges[idx+1]
            if u < v:
                f.write("  edge [\n")
                f.write(f"    source {u}\n")
                f.write(f"    target {v}\n")
                f.write("    weight 1.0\n")
                f.write("  ]\n")
                
        f.write("]\n")
        
    print(f"  GML file written in {time.time()-t0:.1f}s", flush=True)


def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in: {src_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} dataset(s) in {src_dir}")
    print(f"Output GML directory: {dst_dir}\n")

    for fname in txt_files:
        name = fname[:-4]
        gml_path = os.path.join(dst_dir, f"{name}.gml")

        if os.path.exists(gml_path):
            print(f"[SKIP] {name}.gml already exists.\n")
            continue

        src_path = os.path.join(src_dir, fname)
        print(f"[{name}] Starting conversion ...", flush=True)
        t_start = time.time()

        flat_edges, degrees, node_map = parse_and_map_edgelist(src_path)
        if len(degrees) == 0:
            print(f"  [WARN] No valid edges in {fname} – skipping.\n")
            continue

        # PageRank
        pr = compute_pagerank(len(degrees), flat_edges, degrees)

        # Stream to GML file
        stream_to_gml(gml_path, degrees, pr, flat_edges)
        
        print(f"  [DONE] {name}.gml completed in {time.time()-t_start:.1f}s\n", flush=True)

    print("=" * 60)
    print("All conversions complete.")
    print(f"GML files are in: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Social_Networks/*.txt -> Socials/*.gml with degree & pagerank (memory-efficient)"
    )
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="Source folder with .txt edge-lists (default: Social_Networks)")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help="Output GML folder (default: Socials)")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isdir(src):
        print(f"[ERROR] Source directory not found: {src}")
        sys.exit(1)

    convert(src, dst)


if __name__ == "__main__":
    main()
