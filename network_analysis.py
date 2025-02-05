#!/usr/bin/env python3
# Network Analysis Script



import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
import numpy as np

# Function to calculate network properties
def calculate_network_properties(G):
    # Perform community detection using Girvan-Newman
    communities_generator = girvan_newman(G)
    partition = next(communities_generator)

    # Calculate modularity
    modularity = nx.community.modularity(G, partition)

    # Calculate average clustering coefficient
    clustering_coefficient = nx.average_clustering(G)

    # Number of communities
    num_communities = len(partition)

    return modularity, clustering_coefficient, num_communities

# Function to generate and visualize a network with community detection
def visualize_network_with_communities(G, title, filename=None):
    # Perform community detection using Girvan-Newman
    communities_generator = girvan_newman(G)
    partition = next(communities_generator)

    # Create a color map for communities
    community_colors = {}
    for i, community in enumerate(partition):
        for node in community:
            community_colors[node] = i

    # Assign colors to nodes based on their community
    node_colors = [community_colors[node] for node in G.nodes()]

    # Calculate node sizes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [3000 * degree_centrality[node] for node in G.nodes()]

    # Calculate edge widths based on betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    edge_widths = [5 * edge_betweenness[edge] for edge in G.edges()]

    # Assign edge colors based on edge betweenness centrality
    edge_colors = [edge_betweenness[edge] for edge in G.edges()]

    # Draw the network
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)

    # Draw edges with width and color based on betweenness centrality
    edges = nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=0.7
    )

    # Draw nodes with community colors and size based on degree centrality
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.rainbow, alpha=0.8
    )

    # Add labels for high-degree nodes
    high_degree_nodes = [node for node, degree in G.degree() if degree > np.percentile(list(dict(G.degree()).values()), 90)]
    labels = {node: node for node in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="black", font_weight="bold")

    # Add a color bar for communities
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=len(partition) - 1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), label="Community", shrink=0.8, pad=0.02)

    # Add a color bar for edge betweenness centrality
    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=plt.gca(), label="Edge Betweenness Centrality", shrink=0.8, pad=0.02)

    # Add title and annotations
    plt.title(title, fontsize=16, pad=20)
    plt.figtext(
        0.5, 0.02,
        f"Modularity: {nx.community.modularity(G, partition):.3f}, Clustering Coefficient: {nx.average_clustering(G):.3f}, Number of Communities: {len(partition)}",
        ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}
    )

    # Highlight specific communities or nodes (optional)
    for i, community in enumerate(partition):
        if i == 0:  # Highlight the first community
            nx.draw_networkx_nodes(
                G, pos, nodelist=community, node_size=3000, node_color="red", alpha=0.5
            )

    plt.axis("off")  # Turn off axis
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# Function to generate a comparative analysis plot
def generate_comparative_analysis_plot(networks, network_names, filename=None):
    # Calculate properties for each network
    modularities = []
    clustering_coefficients = []
    num_communities = []

    for G in networks:
        modularity, clustering_coefficient, num_community = calculate_network_properties(G)
        modularities.append(modularity)
        clustering_coefficients.append(clustering_coefficient)
        num_communities.append(num_community)

    # Create a grouped bar plot
    x = np.arange(len(network_names))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, modularities, width, label="Modularity", color="skyblue")
    rects2 = ax.bar(x, clustering_coefficients, width, label="Clustering Coefficient", color="lightgreen")
    rects3 = ax.bar(x + width, num_communities, width, label="Number of Communities", color="salmon")

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel("Networks")
    ax.set_ylabel("Values")
    ax.set_title("Comparative Analysis of Network Properties", fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(network_names, rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Add value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom"
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# Load or generate networks
def load_or_generate_networks():
    # Zachary's Karate Club
    zachary = nx.karate_club_graph()

    # Synthetic PPI Network (replace with real data if available)
    ppi = nx.gnm_random_graph(100, 300)  # Smaller size for better visualization

    # Facebook Social Network (replace with real data if available)
    facebook = nx.gnm_random_graph(200, 1000)  # Smaller size for better visualization

    # Email-Eu Core Network (replace with real data if available)
    email_eu = nx.gnm_random_graph(150, 500)  # Smaller size for better visualization

    return zachary, ppi, facebook, email_eu

# Main function to generate and visualize plots
def main():
    # Load or generate networks
    zachary, ppi, facebook, email_eu = load_or_generate_networks()
    networks = [zachary, ppi, facebook, email_eu]
    network_names = ["Zachary's Karate Club", "PPI Network", "Facebook Social Network", "Email-Eu Core Network"]

    # Visualize each network with community detection
    visualize_network_with_communities(zachary, "Zachary's Karate Club Network with Community Detection", "zachary_network.png")
    visualize_network_with_communities(ppi, "Synthetic PPI Network with Community Detection", "ppi_network.png")
    visualize_network_with_communities(facebook, "Facebook Social Network with Community Detection", "facebook_network.png")
    visualize_network_with_communities(email_eu, "Email-Eu Core Network with Community Detection", "email_eu_network.png")

    # Generate comparative analysis plot
    generate_comparative_analysis_plot(networks, network_names, "comparative_analysis.png")

if __name__ == "__main__":
    main()


import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
import numpy as np

# Function to calculate network properties
def calculate_network_properties(G):
    # Perform community detection using Girvan-Newman
    communities_generator = girvan_newman(G)
    partition = next(communities_generator)

    # Calculate modularity
    modularity = nx.community.modularity(G, partition)

    # Calculate average clustering coefficient
    clustering_coefficient = nx.average_clustering(G)

    # Number of communities
    num_communities = len(partition)

    # Average degree
    avg_degree = np.mean([d for n, d in G.degree()])

    return modularity, clustering_coefficient, num_communities, avg_degree

# Function to generate a radar chart for comparative analysis
def generate_radar_chart(networks, network_names, filename=None):
    # Calculate properties for each network
    properties = ["Modularity", "Clustering Coefficient", "Number of Communities", "Average Degree"]
    num_vars = len(properties)

    # Normalize the data for the radar chart
    def normalize_data(data):
        min_val = min(data)
        max_val = max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    # Prepare data for the radar chart
    data = []
    for G in networks:
        modularity, clustering_coefficient, num_communities, avg_degree = calculate_network_properties(G)
        data.append([modularity, clustering_coefficient, num_communities, avg_degree])

    # Normalize the data
    data_normalized = [normalize_data(d) for d in data]

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Plot the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], properties)

    # Draw ylabels
    ax.set_rscale('log')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each network
    colors = ["skyblue", "lightgreen", "salmon", "gold"]
    for i, (values, name) in enumerate(zip(data_normalized, network_names)):
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=name, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    # Add a legend
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    # Add a title
    plt.title("Comparative Analysis of Network Properties", size=16, pad=20)

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# Load or generate networks
def load_or_generate_networks():
    # Zachary's Karate Club
    zachary = nx.karate_club_graph()

    # Synthetic PPI Network (replace with real data if available)
    ppi = nx.gnm_random_graph(100, 300)  # Smaller size for better visualization

    # Facebook Social Network (replace with real data if available)
    facebook = nx.gnm_random_graph(200, 1000)  # Smaller size for better visualization

    # Email-Eu Core Network (replace with real data if available)
    email_eu = nx.gnm_random_graph(150, 500)  # Smaller size for better visualization

    return zachary, ppi, facebook, email_eu

# Main function to generate and visualize plots
def main():
    # Load or generate networks
    zachary, ppi, facebook, email_eu = load_or_generate_networks()
    networks = [zachary, ppi, facebook, email_eu]
    network_names = ["Zachary's Karate Club", "PPI Network", "Facebook Social Network", "Email-Eu Core Network"]

    # Generate radar chart for comparative analysis
    generate_radar_chart(networks, network_names, "comparative_analysis_radar.png")

if __name__ == "__main__":
    main()