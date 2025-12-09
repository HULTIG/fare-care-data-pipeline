import networkx as nx
import matplotlib.pyplot as plt
import os

def generate_nij_graph(output_path):
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes
    nodes = [
        "Race", 
        "Gender", 
        "Age", 
        "Prior Arrests", 
        "Supervision Level", 
        "Employment", 
        "Recidivism"
    ]
    G.add_nodes_from(nodes)

    # Define edges based on domain knowledge and discovery
    edges = [
        ("Age", "Recidivism"),
        ("Prior Arrests", "Recidivism"),
        ("Prior Arrests", "Supervision Level"),
        ("Employment", "Recidivism"),
        ("Race", "Supervision Level"), # The bias path
        ("Supervision Level", "Recidivism"),
        ("Gender", "Recidivism")
    ]
    G.add_edges_from(edges)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Layout
    pos = {
        "Race": (-1, 1),
        "Gender": (-1, -1),
        "Age": (0, 2),
        "Prior Arrests": (0, 1),
        "Supervision Level": (0, 0),
        "Employment": (0, -1),
        "Recidivism": (1, 0)
    }

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black')
    
    # Draw edges
    # Normal edges
    normal_edges = [e for e in edges if e != ("Race", "Supervision Level")]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='black', arrows=True, arrowsize=20)
    
    # Bias edge (Red)
    nx.draw_networkx_edges(G, pos, edgelist=[("Race", "Supervision Level")], edge_color='red', arrows=True, arrowsize=20, width=2)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title("Causal DAG for NIJ Recidivism (Red = Potential Bias Path)")
    plt.axis('off')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    output_file = "results/figures/fig_nij_causal_graph.png"
    # Ensure we are running from root or handle paths correctly. 
    # Assuming running from source root based on previous commands.
    if not os.path.exists("results"):
         # specific handling if run from script dir
         if os.path.exists("../../results"):
             output_file = "../../results/figures/fig_nij_causal_graph.png"
    
    generate_nij_graph(output_file)
