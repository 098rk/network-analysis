import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
from networkx.generators.random_graphs import watts_strogatz_graph, barabasi_albert_graph
from networkx.generators.community import stochastic_block_model

# Function to calculate network properties
def calculate_network_properties(G):
    # Calculate modularity
    communities = list(greedy_modularity_communities(G))
    modularity = nx.community.modularity(G, communities)

    # Calculate average clustering coefficient
    clustering_coefficient = nx.average_clustering(G)

    # Number of communities
    num_communities = len(communities)

    # Average degree
    avg_degree = np.mean([d for n, d in G.degree()])

    return modularity, clustering_coefficient, num_communities, avg_degree

# Function to extract hub nodes (top 10% by degree centrality)
def extract_hub_nodes(G):
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    hub_nodes = [node for node, _ in sorted_nodes[:int(len(G.nodes()) * 0.1)]]
    return hub_nodes

# Function to subsample a large network
def subsample_network(G, target_nodes=1000):
    # Extract hub nodes to preserve important structures
    hub_nodes = extract_hub_nodes(G)

    # Ensure the number of hub nodes does not exceed the target size
    if len(hub_nodes) > target_nodes:
        hub_nodes = hub_nodes[:target_nodes]  # Truncate to target_nodes
        remaining_nodes = []
    else:
        # Randomly sample additional nodes to reach the target size
        remaining_nodes = list(set(G.nodes()) - set(hub_nodes))
        sampled_nodes = np.random.choice(remaining_nodes, size=target_nodes - len(hub_nodes), replace=False)
        hub_nodes = list(hub_nodes) + list(sampled_nodes)

    # Create the subgraph
    subgraph = G.subgraph(hub_nodes)
    return subgraph

# Function to generate and analyze networks
def analyze_networks():
    # Generate synthetic networks
    n = 1000  # Number of nodes
    ws_graph = watts_strogatz_graph(n, k=10, p=0.1)  # Watts-Strogatz small-world model
    ba_graph = barabasi_albert_graph(n, m=5)  # Barabási-Albert scale-free model
    sbm_graph = stochastic_block_model([n // 2, n // 2], [[0.1, 0.01], [0.01, 0.1]])  # SBM

    # Analyze properties
    ws_properties = calculate_network_properties(ws_graph)
    ba_properties = calculate_network_properties(ba_graph)
    sbm_properties = calculate_network_properties(sbm_graph)

    print("Watts-Strogatz Network Properties:", ws_properties)
    print("Barabási-Albert Network Properties:", ba_properties)
    print("Stochastic Block Model Properties:", sbm_properties)

# Function to demonstrate subsampling strategy
def demonstrate_subsampling():
    # Load or generate a large network (e.g., PPI network)
    ppi_graph = nx.gnm_random_graph(19247, 11759712)  # Synthetic PPI network

    # Subsample the network
    subgraph = subsample_network(ppi_graph, target_nodes=1000)

    # Analyze properties of the subgraph
    subgraph_properties = calculate_network_properties(subgraph)
    print("Subsampled PPI Network Properties:", subgraph_properties)

# Main function
def main():
    # Analyze synthetic networks
    analyze_networks()

    # Demonstrate subsampling strategy
    demonstrate_subsampling()

if __name__ == "__main__":
    main()
