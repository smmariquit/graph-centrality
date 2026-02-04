# Graph centrality gradient
# We'll try to simulate a hierarchial network akin to a terrorist organization.

import matplotlib
matplotlib.use('WebAgg')  # Web-based backend - opens in browser

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


def create_hierarchical_network(n_nodes=500, n_hubs=5, n_lieutenants=20, seed=42):
    np.random.seed(seed)
    G = nx.Graph()
    
    # Add nodes 1 to n
    G.add_nodes_from(range(n_nodes))
    
    # Define node categories
    hub_nodes = list(range(n_hubs))  # Top leadership
    lieutenant_nodes = list(range(n_hubs, n_hubs + n_lieutenants))  # Mid-level
    peripheral_nodes = list(range(n_hubs + n_lieutenants, n_nodes))  # Privates
    
    # Connect hubs to each other with a 60% chance
    HUB_CONNECTION_CHANCE = 0.6
    for i in range(len(hub_nodes)):
        for j in range(i + 1, len(hub_nodes)):
            if np.random.random() < HUB_CONNECTION_CHANCE:
                G.add_edge(hub_nodes[i], hub_nodes[j])
    
    # Connect lieutenants to hubs
    for lt in lieutenant_nodes:
        # Each lieutenant connects to 1-3 hubs
        n_hub_connections = np.random.randint(1, min(4, n_hubs + 1))
        connected_hubs = np.random.choice(hub_nodes, n_hub_connections, replace=False)
        for hub in connected_hubs:
            G.add_edge(lt, hub)
        
        # Lieutenants also connect to each other (sparse)
        for other_lt in lieutenant_nodes:
            if other_lt > lt and np.random.random() < 0.15:
                G.add_edge(lt, other_lt)
    
    # Connect peripheral nodes
    for peripheral in peripheral_nodes:
        # Each peripheral connects to 1-3 lieutenants
        n_lt_connections = np.random.randint(1, 4)
        connected_lts = np.random.choice(lieutenant_nodes, n_lt_connections, replace=False)
        for lt in connected_lts:
            G.add_edge(peripheral, lt)
        
        # Small chance of direct hub connection (trusted operatives)
        if np.random.random() < 0.02:
            hub = np.random.choice(hub_nodes)
            G.add_edge(peripheral, hub)
        
        # Peripheral to peripheral connections (cells)
        for other_peripheral in peripheral_nodes:
            if other_peripheral > peripheral and np.random.random() < 0.005:
                G.add_edge(peripheral, other_peripheral)
    
    return G, hub_nodes, lieutenant_nodes, peripheral_nodes

# You can adjust the centrality measure as needed
# Other measures include: degree_centrality, closeness_centrality, eigenvector_centrality
# Centrality in networkx is a value between 0 and 1
def calculate_centrality(G):
    centrality = nx.betweenness_centrality(G)
    return centrality

# Create a color gradient for edges based on centrality of connected nodes
# Returns a list of colors for each edge
def get_edge_gradient_colors(G, centrality, pos):
    edges = list(G.edges())
    edge_colors = []
    
    # Normalize centrality values with center at 0.5
    max_cent = max(centrality.values())
    min_cent = min(centrality.values())
    
    if max_cent == min_cent:
        norm_centrality = {k: 0.5 for k in centrality}
    else:
        norm_centrality = {k: (v - min_cent) / (max_cent - min_cent) for k, v in centrality.items()}
    
    for u, v in edges:
        # Get centrality of both endpoints
        cent_u = norm_centrality[u]
        cent_v = norm_centrality[v]
        
        # Create color based on average centrality of edge endpoints
        # Higher centrality = darker red (lower RGB values for green and blue)
        avg_cent = (cent_u + cent_v) / 2
        
        # Dark red to light red (pink/white)
        # High centrality: dark red (0.6, 0, 0)
        # Low centrality: light red/pink (1.0, 0.8, 0.8)
        r = 0.7 + 0.3 * (1 - avg_cent)
        g = 0.8 * (1 - avg_cent)
        b = 0.8 * (1 - avg_cent)
        
        edge_colors.append((r, g, b, 0.6))  # Add alpha for transparency
    
    return edge_colors


def draw_gradient_edges(G, pos, centrality, ax):
    # Normalize centrality
    max_cent = max(centrality.values())
    min_cent = min(centrality.values())
    
    if max_cent == min_cent:
        norm_centrality = {k: 0.5 for k in centrality}
    else:
        norm_centrality = {k: (v - min_cent) / (max_cent - min_cent) for k, v in centrality.items()}
    
    # Create colormap from dark red to light pink
    colors_list = ['#8B0000', '#CD5C5C', '#F08080', '#FFB6C1', '#FFE4E1']
    cmap = mcolors.LinearSegmentedColormap.from_list('red_gradient', colors_list)
    
    for u, v in G.edges():
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        
        # Create multiple segments for gradient effect
        n_segments = 20
        
        cent_u = norm_centrality[u]
        cent_v = norm_centrality[v]
        
        # Create line segments
        points = np.array([
            np.linspace(x_coords[0], x_coords[1], n_segments + 1),
            np.linspace(y_coords[0], y_coords[1], n_segments + 1)
        ]).T.reshape(-1, 1, 2)
        
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create color values that transition from u's centrality to v's centrality
        # Invert so high centrality = dark (low value in reversed colormap)
        color_values = np.linspace(1 - cent_u, 1 - cent_v, n_segments)
        
        # Create linewidth values: thicker near high centrality nodes
        # High centrality = 3.0, Low centrality = 0.3
        min_width = 0.3
        max_width = 3.0
        linewidths = np.linspace(
            min_width + (max_width - min_width) * cent_u,
            min_width + (max_width - min_width) * cent_v,
            n_segments
        )
        
        lc = LineCollection(segments, cmap=cmap, alpha=0.6)
        lc.set_array(color_values)
        lc.set_linewidths(linewidths)
        ax.add_collection(lc)


def visualize_network(G, centrality, hub_nodes, lieutenant_nodes, peripheral_nodes):
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Use Kamada-Kawai layout which works without scipy for small graphs
    # or fall back to basic spring layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        # Fallback to basic spring layout without scipy optimization
        pos = nx.random_layout(G, seed=42)
        # Manual spring layout iterations
        for _ in range(50):
            for node in G.nodes():
                if G.degree(node) > 0:
                    neighbors = list(G.neighbors(node))
                    avg_x = sum(pos[n][0] for n in neighbors) / len(neighbors)
                    avg_y = sum(pos[n][1] for n in neighbors) / len(neighbors)
                    pos[node] = (
                        pos[node][0] * 0.5 + avg_x * 0.5,
                        pos[node][1] * 0.5 + avg_y * 0.5
                    )
    
    # Draw gradient edges
    print("Drawing gradient edges...")
    draw_gradient_edges(G, pos, centrality, ax)
    
    # Normalize centrality for node sizing
    max_cent = max(centrality.values())
    min_cent = min(centrality.values())
    
    if max_cent == min_cent:
        norm_centrality = {k: 0.5 for k in centrality}
    else:
        norm_centrality = {k: (v - min_cent) / (max_cent - min_cent) for k, v in centrality.items()}
    
    # Node sizes based on centrality
    node_sizes = [100 + 1500 * norm_centrality[node] for node in G.nodes()]
    
    # Node colors based on category
    node_colors = []
    for node in G.nodes():
        if node in hub_nodes:
            node_colors.append('#ffffff')  # Dark red for hubs
        elif node in lieutenant_nodes:
            node_colors.append('#ffffff')  # Indian red for lieutenants
        else:
            node_colors.append('#ffffff')  # Light pink for peripherals
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add labels only for hub nodes
    hub_labels = {node: f'Hub {node}' for node in hub_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=hub_labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Create colorbar for centrality
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list(
            'red_gradient', 
            ['#8B0000', '#CD5C5C', '#F08080', '#FFB6C1', '#FFE4E1']
        ),
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Betweenness Centrality')
    cbar.ax.set_yticklabels(['High', '', '', '', 'Low'])
    
    ax.set_title(
        'Hierarchical Network Visualization (500 Nodes)\n'
        'Edge Gradient: Dark Red (High Centrality) → Light Pink (Low Centrality)',
        fontsize=14,
        fontweight='bold'
    )
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#8B0000', s=200, label=f'Hub Nodes ({len(hub_nodes)})', edgecolors='black'),
        plt.scatter([], [], c='#CD5C5C', s=100, label=f'Lieutenant Nodes ({len(lieutenant_nodes)})', edgecolors='black'),
        plt.scatter([], [], c='#FFB6C1', s=50, label=f'Peripheral Nodes ({len(peripheral_nodes)})', edgecolors='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def print_network_stats(G, centrality, hub_nodes, lieutenant_nodes):
    """Print network statistics."""
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Network density: {nx.density(G):.4f}")
    
    # Top 10 nodes by centrality
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 nodes by Betweenness Centrality:")
    print("-" * 40)
    for i, (node, cent) in enumerate(sorted_centrality[:10], 1):
        node_type = "Hub" if node in hub_nodes else ("Lieutenant" if node in lieutenant_nodes else "Peripheral")
        print(f"  {i}. Node {node} ({node_type}): {cent:.4f}")
    
    print("="*60 + "\n")


def main():
    print("Creating hierarchical network with 500 nodes...")
    G, hub_nodes, lieutenant_nodes, peripheral_nodes = create_hierarchical_network(
        n_nodes=500,
        n_hubs=5,
        n_lieutenants=20,
        seed=42
    )
    
    print("Calculating betweenness centrality...")
    centrality = calculate_centrality(G)
    
    # Print statistics
    print_network_stats(G, centrality, hub_nodes, lieutenant_nodes)
    
    print("Creating visualization with edge gradients...")
    fig = visualize_network(G, centrality, hub_nodes, lieutenant_nodes, peripheral_nodes)
    
    # Save the figure
    output_path = '/workspaces/graph-centrality/network_visualization.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    # Open interactive window - block=True keeps window open
    print("Opening interactive window...")
    plt.show(block=True)


if __name__ == "__main__":
    main()
