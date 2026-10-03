import os
import glob
import networkx as nx
import xml.etree.ElementTree as ET

def main():
    input_dir = "Social_Network"
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
        
    out_dir = "NodeAttributes_XML"
    os.makedirs(out_dir, exist_ok=True)
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)
        out_filepath = os.path.join(out_dir, f"{name}.xml")
        
        if os.path.exists(out_filepath):
            print(f"Skipping {name}, already processed ({out_filepath} exists).")
            continue
            
        print(f"Processing {name} from {filepath}...")
        
        # Load the graph as undirected (nx.Graph) to match run_social.py's symmetric edge handling.
        # Use data=False to ignore any extra columns (like timestamps or weights) which cause parsing errors.
        G = nx.read_edgelist(filepath, comments='#', create_using=nx.Graph(), nodetype=int, data=False)
        
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        print("  Computing PageRank...")
        pr = nx.pagerank(G, alpha=0.85)
        
        print("  Computing Degree...")
        # Since it's an undirected graph, G.degree() natively returns the correct total degree
        degrees = dict(G.degree())
        
        print(f"  Saving node attributes to {out_filepath} (XML format)...")
        # Build custom clean XML with just node attributes as requested
        root = ET.Element("dataset", name=name)
        nodes_elem = ET.SubElement(root, "nodes")
        
        for node in sorted(G.nodes()):
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.set("id", str(node))
            node_elem.set("degree", str(degrees[node]))
            # Format pagerank cleanly
            node_elem.set("pagerank", f"{pr[node]:.6g}")
            
        tree = ET.ElementTree(root)
        # Indent for readability (available in Python 3.9+)
        if hasattr(ET, 'indent'):
            ET.indent(tree, space="  ", level=0)
            
        tree.write(out_filepath, encoding="utf-8", xml_declaration=True)
        print("  Done.\n")

if __name__ == "__main__":
    main()
