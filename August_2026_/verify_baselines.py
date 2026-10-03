import os
import sys
import networkx as nx

from myopic import myopic
from naive_myopic import naive_myopic
from gonzalez import gonzalez
from icm import prob_est_timed

def compute_variance(probs):
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def main():
    print("=== VERIFYING MYOPIC/NAIVE/GONZALEZ ON FACEBOOK ALPHA=0.03, K=5 ===")
    
    gml_path = os.path.join("Social_Network", "Facebook.gml")
    G = nx.read_gml(gml_path, destringizer=int)
    original_nodes = sorted([int(n) for n in G.nodes()])
    id_map = {orig: new for new, orig in enumerate(original_nodes)}
    n = len(original_nodes)
    
    nodes = [{} for _ in range(n)]
    for n_id, attrs in G.nodes(data=True):
        mapped_id = id_map[int(n_id)]
        nodes[mapped_id] = attrs
        
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        u, v = int(u), int(v)
        mu, mv = id_map[u], id_map[v]
        adj[mu].append(mv)
        adj[mv].append(mu)
        
    alpha = 0.03
    k = 5
    T = 15
    R = 200
    
    print("\n--- Running Myopic Baseline ---")
    m_seeds, m_probs, m_hits = myopic(adj, nodes, alpha, k, T, R)
    print("Myopic Seeds:", m_seeds)
    print("Myopic Variance:", compute_variance(m_probs))
    print("Myopic Mean Prob:", sum(m_probs)/n)
    
    print("\n--- Running Naive Myopic Baseline ---")
    nm_seeds, nm_probs, nm_hits = naive_myopic(adj, nodes, alpha, k, T, R)
    print("Naive Myopic Seeds:", nm_seeds)
    print("Naive Myopic Variance:", compute_variance(nm_probs))
    print("Naive Myopic Mean Prob:", sum(nm_probs)/n)
    
    print("\n--- Running Gonzalez Baseline ---")
    g_seeds, g_probs, g_hits = gonzalez(adj, nodes, alpha, k, T, R)
    print("Gonzalez Seeds:", g_seeds)
    print("Gonzalez Variance:", compute_variance(g_probs))
    print("Gonzalez Mean Prob:", sum(g_probs)/n)

if __name__ == "__main__":
    main()
