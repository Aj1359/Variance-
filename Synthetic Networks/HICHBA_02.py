import random
'''import seaborn as sns'''
import networkx as nx
from tqdm import tqdm

'''import matplotlib.pyplot as plt'''

"""
HICH-BA code from: https://github.com/akratiiet/FWRRS/blob/main/HICH-BA.ipynb

The HICH-BA model uses the following parameters: 
(i) n, i.e., the desired number of nodes,
(ii) p_N , i.e., the probability of adding a node to the graph, with probability 1 − pN an edge is added,
(iii) r, list where each entry ri corresponds to the probability of a new node belonging to community i,
(iv) h, i.e., the homophily factor and represents the probability of a node establishing an intra-community connection,
(v)p_T, the probability to form a close triad connection,
(vi) p_PA, the probability with which a new edge will be established using the preferential attachment (PA)
"""
def hichba(n,r,h,p_PA,p_N,p_T):
    
    num_com=len(r)
    G= nx.Graph()
    nx.set_node_attributes(G, [], "ground_truth")
    G.add_nodes_from(range(num_com))
    nodes=len(G.nodes())
    
    choices_c={c:[] for c in range(num_com)}
    choices_weights_c={c:{} for c in range(num_com)}
    
    c=0
    for v in G.nodes():
        G.nodes[v]['ground_truth']= c
        choices_c[c].append(v)
        choices_weights_c[c][v]=1
        c+=1
        
    L_values,x_val=[],[]
    pbar = tqdm(total=n, position=0, leave=True)
    pbar.update(len(G.nodes()))
    h_orig=h
    while nodes<=n:
        if random.uniform(0,1)<=p_N:
            G.add_node(nodes-1)
            source=nodes-1
            nodes+=1
            c=random.choices(range(num_com), weights=r, k=1)[0]
            G.nodes[source]['ground_truth']= c

            choices_c[c].append(source)
            choices_weights_c[c][source]=1 

            choices=[x for x in choices_c[c] if x!=source]

            if random.uniform(0, 1)<=(1-p_PA):weights=[1 for v in choices]
            else:weights=[choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices]
            
            
            if len(choices)==0:continue
            target=random.choices(choices, weights=weights, k=1)[0]

            G.add_edge(source, target)

            choices_weights_c[c][source]+=1
            choices_weights_c[G.nodes[target]['ground_truth']][target]+=1
            pbar.update(1)

        else:
            if random.uniform(0,1)<=p_T:
                if random.uniform(0,1)<=(1-p_PA):
                    if len([x for x in G.nodes() if G.degree(x)>=2])==0:continue
                    v=random.choice([x for x in G.nodes() if G.degree(x)>=2])
                else:
                    if len([x for x in G.nodes() if G.degree(x)>=2])==0:continue
                    v=random.choices([x for x in G.nodes() if G.degree(x)>=2], weights=[G.degree(x)+1 for x in G.nodes() if G.degree(x)>=2],k=1)[0]
                
                target1=random.choice(list(G.neighbors(v)))
                options=[y for y in G.neighbors(v) if not G.has_edge(target1,y)]
                if len(options)==0: continue
                intra_inter= random.uniform(0, 1)
                if intra_inter<=h: choices=[x for x in options if G.nodes[v]['ground_truth']==G.nodes[x]['ground_truth']]
                else:choices=[x for x in options if G.nodes[v]['ground_truth']!=G.nodes[x]['ground_truth']]
                    
                if random.uniform(0, 1)<=(1-p_PA):weights=[1 for w in options]
                else: weights=[choices_weights_c[G.nodes[w]['ground_truth']][w] for w in options] 
                
                if len(options)==0: print("no ", intra_inter);continue
                target2=random.choices(options, weights=weights, k=1)[0]
                
                G.add_edge(target1, target2)
                choices_weights_c[G.nodes[target1]['ground_truth']][target1]+=1
                choices_weights_c[G.nodes[target2]['ground_truth']][target2]+=1
                
                
            else:
                if random.uniform(0,1)<=(1-p_PA):
                    v=random.choice([x for x in G.nodes() ])
                else:
                    v=random.choices([x for x in G.nodes() ], weights=[G.degree(x)+1 for x in G.nodes()],k=1)[0]
                    
                neigh=list( G.neighbors(v))
                options=[x for x in G.nodes() if x not in neigh]
                intra_inter= random.uniform(0, 1)
                if intra_inter<=h: choices=[x for x in options if G.nodes[v]['ground_truth']==G.nodes[x]['ground_truth']]
                else:choices=[x for x in options if G.nodes[v]['ground_truth']!=G.nodes[x]['ground_truth']]

                if random.uniform(0, 1)<=(1-p_PA):weights=[1 for v in choices]
                else:weights=[choices_weights_c[G.nodes[v]['ground_truth']][v] for v in choices] 

                if len(choices)==0:continue
                target=random.choices(choices, weights=weights, k=1)[0]
                if (intra_inter>h and random.uniform(0,1)<=r[G.nodes[target]['ground_truth']]/r[G.nodes[v]['ground_truth']]) or intra_inter<h :
                    G.add_edge(v, target)

                    choices_weights_c[G.nodes[v]['ground_truth']][v]+=1
                    choices_weights_c[G.nodes[target]['ground_truth']][target]+=1
        
    return G



G = hichba(n=10000, r=[0.5, 0.3, 0.15, 0.05], h=0.2, p_PA=0.7, p_N=1/10, p_T=0.3)
nx.write_gml(G, "HICHBA_02.gml")