import time
from icm import prob_est_timed

def myopic(adj, nodes, alpha, k, T, R=200):
    """
    Standard Myopic Baseline:
    Picks the node with the absolute lowest probability of being informed.
    Tie-breaks using highest degree.
    """
    n = len(nodes)
    seeds = []
    
    for step in range(k):
        # Estimate probabilities
        probs, hits = prob_est_timed(adj, seeds, alpha, n, T, R)
        
        best_cand = None
        min_p = float('inf')
        best_deg = -1
        
        for i in range(n):
            if i in seeds: continue
            
            p = probs[i]
            deg = len(adj[i])
            
            # Minimize probability, tie-break by highest degree
            if p < min_p:
                min_p = p
                best_deg = deg
                best_cand = i
            elif p == min_p:
                if deg > best_deg:
                    best_deg = deg
                    best_cand = i
                    
        seeds.append(best_cand)
        
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits
