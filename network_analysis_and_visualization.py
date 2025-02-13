import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Set global style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)


def plot_network(graph, title, node_size=50, edge_width=0.5):
    """
    Plot a network with community detection and betweenness centrality using only networkx.
    """
    # Ensure the graph is undirected
    if graph.is_directed():
        graph = graph.to_undirected()

    # Detect communities using the greedy modularity maximization algorithm
    communities = nx.community.greedy_modularity_communities(graph)
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [community_map[node] for node in graph.nodes()]

    # Compute betweenness centrality for edges
    edge_betweenness = nx.edge_betweenness_centrality(graph)

    # Handle missing edges in edge_betweenness
    edge_weights = []
    for u, v in graph.edges():
        if (u, v) in edge_betweenness:
            edge_weights.append(edge_betweenness[(u, v)] * 5)
        elif (v, u) in edge_betweenness:  # Check reverse edge for undirected graphs
            edge_weights.append(edge_betweenness[(v, u)] * 5)
        else:
            edge_weights.append(0)  # Default weight for missing edges

    # Use a spring layout for better visualization
    pos = nx.spring_layout(graph, seed=42)

    # Plot nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_colors, cmap=plt.cm.tab20, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, width=edge_width, edge_color=edge_weights, edge_cmap=plt.cm.Blues, alpha=0.7)

    # Add title and remove axes
    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.show()


# Zachary’s Karate Club
karate = nx.karate_club_graph()
plot_network(karate, "Zachary’s Karate Club Network", node_size=500)

# PPI Network (STRING)
# Create a synthetic large PPI network for demonstration
ppi = nx.scale_free_graph(n=1000, alpha=0.5, beta=0.3, gamma=0.2)
ppi = ppi.to_undirected()  # Ensure the graph is undirected
plot_network(ppi, "PPI Network (STRING)", node_size=20, edge_width=0.2)

# Facebook Social Network
# Create a synthetic Facebook network for demonstration
facebook = nx.scale_free_graph(n=2000, alpha=0.6, beta=0.3, gamma=0.1)
facebook = facebook.to_undirected()  # Ensure the graph is undirected
plot_network(facebook, "Facebook Social Network", node_size=10, edge_width=0.1)

# Email-Eu Core Network
# Create a synthetic Email-Eu Core network for demonstration
email = nx.scale_free_graph(n=1000, alpha=0.4, beta=0.3, gamma=0.3)
email = email.to_undirected()  # Ensure the graph is undirected
plot_network(email, "Email-Eu Core Network", node_size=20, edge_width=0.2)
