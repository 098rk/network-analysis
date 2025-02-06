import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import modularity
from networkx.generators.community import stochastic_block_model

# Function to generate SBM networks and calculate modularity
def calculate_sbm_modularity(n, p_intra, p_inter):
    # Define the community structure
    sizes = [n // 2, n // 2]  # Two communities of equal size
    probs = [[p_intra, p_inter], [p_inter, p_intra]]  # Intra- and inter-community probabilities

    # Generate the SBM network
    G = stochastic_block_model(sizes, probs, seed=42)

    # Detect communities (ground truth is known for SBM)
    communities = [set(range(n // 2)), set(range(n // 2, n))]

    # Calculate modularity
    Q = modularity(G, communities)
    return Q

# Function to generate heatmaps for SBM parameter sensitivity
def generate_sbm_heatmaps():
    # Network sizes to analyze
    network_sizes = [100, 500, 1000]

    # Range of intra- and inter-community probabilities
    p_intra_values = np.linspace(0.1, 0.9, 20)  # Intra-community probabilities
    p_inter_values = np.linspace(0.01, 0.5, 20)  # Inter-community probabilities

    # Create a figure for the heatmaps
    fig, axes = plt.subplots(1, len(network_sizes), figsize=(18, 6))

    for i, n in enumerate(network_sizes):
        # Initialize a matrix to store modularity values
        modularity_matrix = np.zeros((len(p_intra_values), len(p_inter_values)))

        # Calculate modularity for each combination of p_intra and p_inter
        for j, p_intra in enumerate(p_intra_values):
            for k, p_inter in enumerate(p_inter_values):
                modularity_matrix[j, k] = calculate_sbm_modularity(n, p_intra, p_inter)

        # Plot the heatmap
        ax = axes[i]
        im = ax.imshow(modularity_matrix, cmap="viridis", origin="lower",
                       extent=[p_inter_values[0], p_inter_values[-1], p_intra_values[0], p_intra_values[-1]],
                       aspect="auto")
        ax.set_xlabel("Inter-Community Probability ($p_{inter}$)")
        ax.set_ylabel("Intra-Community Probability ($p_{intra}$)")
        ax.set_title(f"SBM Parameter Sensitivity (n = {n})")
        fig.colorbar(im, ax=ax, label="Modularity (Q)")

        # Add a dashed red line for the modularity threshold (Q = 0.3)
        ax.contour(p_inter_values, p_intra_values, modularity_matrix, levels=[0.3], colors="red", linestyles="dashed")

    plt.tight_layout()
    plt.savefig("sbm_parameter_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.show()

# Main function to generate and visualize heatmaps
def main():
    generate_sbm_heatmaps()

if __name__ == "__main__":
    main()
