import os
import sys
import networkx as nx
from concave_hybrid import concave_hybrid
from myopic_hybrid_kcore import myopic_hybrid_kcore

def main():
    print("Testing concave_hybrid return format...")
    # Build a tiny graph
    G = nx.erdos_renyi_graph(20, 0.3)
    n = 20
    adj = [list(G.neighbors(i)) for i in range(n)]
    nodes = [{"id": i, "group": 0} for i in range(n)]
    
    alpha = 0.3
    k = 3
    T = 15
    R = 10
    
    try:
        res = concave_hybrid(adj, nodes, alpha, k, T, R=R, R_probe=2, lambda_=1.0, verbose=True)
        print("concave_hybrid returned type:", type(res))
        print("concave_hybrid returned length:", len(res))
        for idx, item in enumerate(res):
            print(f"  Item {idx} type: {type(item)}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
