"""
attempt2/graph_loader.py
========================
Loads HICHBA homophily graphs (h=0.2,0.4,0.6,0.8) from .gml files
into the standard (nodes, adj, edges) format used by all algorithm files.

For very large graphs (n > 2000) we extract a subgraph via BFS from the
highest-degree node so experiments remain tractable.  The subgraph preserves
group membership and community structure.

GML format expected:
    node [ id 0  ground_truth 0 ]
    edge [ source 0  target 1 ]
"""

import os
import re
from collections import deque


# ---------------------------------------------------------------------------
# Core GML parser (no NetworkX dependency)
# ---------------------------------------------------------------------------

def load_gml_file(path):
    """
    Parse a HICHBA .gml file.

    Returns
    -------
    nodes : list[dict]   {id:int, group:int, label:str}
    adj   : list[set]    0-indexed adjacency list
    edges : list[tuple]  (u, v) undirected
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # -- nodes --
    node_blocks = re.findall(r'node\s*\[(.*?)\]', text, re.DOTALL)
    raw_nodes = {}
    for block in node_blocks:
        nid   = re.search(r'\bid\s+(\d+)', block)
        label = re.search(r'label\s+"?([^"\s\]]+)"?', block)
        group = re.search(
            r'(?:ground_truth|group|value|community|color)\s+(\d+)', block)
        if nid:
            i = int(nid.group(1))
            raw_nodes[i] = {
                "id":    i,
                "label": label.group(1) if label else str(i),
                "group": int(group.group(1)) if group else 0,
            }

    sorted_ids = sorted(raw_nodes.keys())
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    nodes = []
    for new_i, old in enumerate(sorted_ids):
        nd = raw_nodes[old].copy()
        nd["id"] = new_i
        nodes.append(nd)
    n = len(nodes)

    # -- edges --
    edge_blocks = re.findall(r'edge\s*\[(.*?)\]', text, re.DOTALL)
    edge_set = set()
    for block in edge_blocks:
        src = re.search(r'source\s+(\d+)', block)
        tgt = re.search(r'target\s+(\d+)', block)
        if src and tgt:
            u = id_map.get(int(src.group(1)))
            v = id_map.get(int(tgt.group(1)))
            if u is not None and v is not None and u != v:
                edge_set.add((min(u, v), max(u, v)))

    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    return nodes, adj, list(edge_set)


# ---------------------------------------------------------------------------
# LCC extractor
# ---------------------------------------------------------------------------

def largest_connected_component(nodes, adj, edges):
    """Extract and re-index the largest connected component."""
    n = len(nodes)
    visited = [False] * n
    best_comp = []

    for start in range(n):
        if visited[start]:
            continue
        comp = []
        q = deque([start])
        visited[start] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    q.append(u)
        if len(comp) > len(best_comp):
            best_comp = comp

    lcc_set = set(best_comp)
    old2new = {old: new for new, old in enumerate(sorted(best_comp))}

    new_nodes = []
    for old in sorted(best_comp):
        nd = nodes[old].copy()
        nd["id"] = old2new[old]
        new_nodes.append(nd)

    n2 = len(new_nodes)
    new_adj = [set() for _ in range(n2)]
    new_edges = []
    seen = set()
    for u, v in edges:
        if u in lcc_set and v in lcc_set:
            nu, nv = old2new[u], old2new[v]
            key = (min(nu, nv), max(nu, nv))
            if key not in seen:
                seen.add(key)
                new_edges.append(key)
                new_adj[nu].add(nv)
                new_adj[nv].add(nu)

    return new_nodes, new_adj, new_edges


# ---------------------------------------------------------------------------
# BFS subgraph sampler (for large graphs)
# ---------------------------------------------------------------------------

def bfs_subgraph(nodes, adj, edges, max_n, seed_node=None):
    """
    Stratified multi-seed BFS subgraph sampler.

    Samples proportionally from each demographic group so the subgraph
    preserves the community structure of the full HICHBA graph.
    Starting purely from the highest-degree hub produces a dense clique
    (avg_deg ~ n/5) that saturates all algorithms — useless for comparison.

    Strategy:
      1. Pick one random seed per group (biased toward medium-degree nodes
         so we avoid hubs that collapse everyone into one clique).
      2. Run BFS round-robin across group seeds, collecting max_n/4 nodes
         per group (then fill remaining budget from leftover BFS frontier).
    """
    import random
    n = len(nodes)
    if n <= max_n:
        return nodes, adj, edges

    # Build group membership
    gm = {}
    for i, nd in enumerate(nodes):
        g = nd["group"]
        gm.setdefault(g, [])
        gm[g].append(i)

    # Per-group quota (proportional to group size)
    total_groups = len(gm)
    quota_per_group = max(max_n // total_groups, 5)

    # Pick seeds: median-degree node per group (not hub, not leaf)
    rng = random.Random(42)
    seeds_per_group = {}
    for g, members in gm.items():
        by_deg = sorted(members, key=lambda i: len(adj[i]))
        mid    = len(by_deg) // 2
        seeds_per_group[g] = by_deg[mid]

    # Round-robin BFS: collect quota from each group
    visited    = set()
    group_queues = {}
    group_counts = {g: 0 for g in gm}

    for g, s in seeds_per_group.items():
        visited.add(s)
        group_queues[g] = deque([s])

    collected = []
    groups_cycle = list(sorted(gm.keys()))

    while len(collected) < max_n:
        progress = False
        for g in groups_cycle:
            if group_counts[g] >= quota_per_group:
                continue
            q = group_queues[g]
            if not q:
                continue
            v = q.popleft()
            collected.append(v)
            group_counts[g] += 1
            progress = True
            # Shuffle neighbours to avoid always picking highest-degree next
            nbrs = list(adj[v])
            rng.shuffle(nbrs)
            for u in nbrs:
                if u not in visited:
                    visited.add(u)
                    # Route u to its own group's queue
                    u_group = nodes[u]["group"]
                    group_queues.setdefault(u_group, deque())
                    group_queues[u_group].append(u)
            if len(collected) >= max_n:
                break
        if not progress:
            # All quotas full — fill remaining budget from any non-empty queue
            for g in groups_cycle:
                q = group_queues[g]
                while q and len(collected) < max_n:
                    v = q.popleft()
                    if v not in set(collected):
                        collected.append(v)
            break

    keep    = set(collected[:max_n])
    old2new = {old: new for new, old in enumerate(sorted(keep))}

    sub_nodes = []
    for old in sorted(keep):
        nd = nodes[old].copy()
        nd["id"] = old2new[old]
        sub_nodes.append(nd)

    n2      = len(sub_nodes)
    sub_adj = [set() for _ in range(n2)]
    sub_edges = []
    seen_e  = set()
    for u, v in edges:
        if u in keep and v in keep:
            nu, nv = old2new[u], old2new[v]
            key    = (min(nu, nv), max(nu, nv))
            if key not in seen_e:
                seen_e.add(key)
                sub_edges.append(key)
                sub_adj[nu].add(nv)
                sub_adj[nv].add(nu)

    return sub_nodes, sub_adj, sub_edges


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_all_hichba(folder, max_n=2000, verbose=True):
    """
    Load all HICHBA_*.gml graphs from folder.

    Parameters
    ----------
    folder : str   path containing HICHBA_02.gml, HICHBA_04.gml, etc.
    max_n  : int   subsample to at most this many nodes (0 = no limit)

    Returns
    -------
    dict: {h_float: (nodes, adj, edges)}
    """
    networks = {}
    pattern = re.compile(r'HICHBA_(\d+)\.gml$', re.IGNORECASE)

    for fname in sorted(os.listdir(folder)):
        m = pattern.match(fname)
        if not m:
            continue
        h = int(m.group(1)) / 10.0
        path = os.path.join(folder, fname)

        if verbose:
            print(f"  Loading {fname}  (h={h}) ...", end=" ", flush=True)

        nodes, adj, edges = load_gml_file(path)
        nodes, adj, edges = largest_connected_component(nodes, adj, edges)

        n_full = len(nodes)
        if max_n and n_full > max_n:
            nodes, adj, edges = bfs_subgraph(nodes, adj, edges, max_n)
            if verbose:
                print(f"subsampled {n_full}->{len(nodes)} nodes  "
                      f"{len(edges)} edges  "
                      f"groups={len(set(nd['group'] for nd in nodes))}")
        else:
            if verbose:
                print(f"{len(nodes)} nodes  {len(edges)} edges  "
                      f"groups={len(set(nd['group'] for nd in nodes))}")

        networks[h] = (nodes, adj, edges)

    return networks


def graph_summary(nodes, adj, edges):
    n = len(nodes)
    degs = [len(a) for a in adj]
    return {
        "n":          n,
        "edges":      len(edges),
        "groups":     len(set(nd["group"] for nd in nodes)),
        "avg_degree": round(sum(degs) / n, 2) if n else 0,
        "max_degree": max(degs) if degs else 0,
    }
