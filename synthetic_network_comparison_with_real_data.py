import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import procrustes


# Generate synthetic networks
def generate_network(model_type, **params):
    if model_type == "ER":
        return nx.erdos_renyi_graph(params['n'], params['p'])
    elif model_type == "BA":
        return nx.barabasi_albert_graph(params['n'], params['m'])
    elif model_type == "SBM":
        sizes = params['sizes']
        p_in = params['p_in']  # Intra-community probability
        p_out = params['p_out']  # Inter-community probability
        # Create a probability matrix for SBM
        prob_matrix = np.full((len(sizes), len(sizes)), p_out)
        np.fill_diagonal(prob_matrix, p_in)  # Set intra-community probabilities
        return nx.stochastic_block_model(sizes, prob_matrix)
    elif model_type == "WS":
        return nx.watts_strogatz_graph(params['n'], params['k'], params['p'])
    elif model_type == "Multilayer":
        return generate_multilayer_network(params['n'], params['m'])
    else:
        raise ValueError("Unknown model type")


# Generate Multilayer Network (simple model for demonstration)
def generate_multilayer_network(n, m):
    G = nx.erdos_renyi_graph(n, 0.1)
    return G


# Generate synthetic models
synthetic_models = {
    "ER": generate_network("ER", n=1000, p=0.05),
    "BA": generate_network("BA", n=1000, m=5),
    "SBM": generate_network("SBM", sizes=[500, 500], p_in=0.1, p_out=0.01),  # Fixed sizes to match total nodes
    "WS": generate_network("WS", n=1000, k=6, p=0.1),
    "Multilayer": generate_multilayer_network(n=1000, m=5)
}

# Real-world network (example: Zachary Karate Club)
real_network = nx.karate_club_graph()


# Function to calculate degree distribution (for simplicity)
def degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    return np.histogram(degree_sequence, bins=30, density=True)


# Compare synthetic and real-world networks using Procrustes analysis
def procrustes_analysis(synthetic_data, real_data):
    # Ensure both inputs are 2D arrays (by reshaping the 1D data)
    synthetic_data_2d = np.reshape(synthetic_data, (-1, 1))
    real_data_2d = np.reshape(real_data, (-1, 1))

    # Align synthetic data to real data using Procrustes
    _, aligned_synthetic_data, _ = procrustes(real_data_2d, synthetic_data_2d)
    return aligned_synthetic_data


# Bootstrap method to calculate confidence intervals
def bootstrap_confidence_interval(data, num_resamples=1000, alpha=0.05):
    # Flatten the data to ensure it is 1-dimensional
    data = data.flatten()

    resampled_means = []
    for _ in range(num_resamples):
        resample = np.random.choice(data, size=len(data), replace=True)
        resampled_means.append(np.mean(resample))
    lower = np.percentile(resampled_means, 100 * alpha / 2)
    upper = np.percentile(resampled_means, 100 * (1 - alpha / 2))
    return lower, upper


# Collect data for comparison
synthetic_data = []
real_data = degree_distribution(real_network)[0]

# Ensure the degree distributions are of the same length by trimming or padding
num_bins = len(real_data)
for model_name, G in synthetic_models.items():
    synth_data = degree_distribution(G)[0]
    # If synthetic data is longer, trim it, if shorter, pad it
    if len(synth_data) < num_bins:
        synth_data = np.pad(synth_data, (0, num_bins - len(synth_data)), 'constant', constant_values=0)
    elif len(synth_data) > num_bins:
        synth_data = synth_data[:num_bins]

    synthetic_data.append(synth_data)

# Perform Procrustes analysis for each model
aligned_synthetic_data = [procrustes_analysis(synth_data, real_data) for synth_data in synthetic_data]

# Plot results
plt.figure(figsize=(10, 6))

# Plot real data
plt.plot(real_data, label="Real-world network", color='black', linewidth=2)

# Plot each synthetic model
labels = list(synthetic_models.keys())
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, aligned_data in enumerate(aligned_synthetic_data):
    # Compute confidence intervals using bootstrap
    lower, upper = bootstrap_confidence_interval(aligned_data)

    # Plot the shaded confidence interval region
    plt.fill_between(range(len(aligned_data)), lower, upper, color=colors[i], alpha=0.2)

    # Plot the aligned synthetic data
    plt.plot(aligned_data, label=f"Synthetic {labels[i]}", color=colors[i])

# Set plot details
plt.xlabel('Degree')
plt.ylabel('Density')
plt.title('Comparison of Synthetic and Real-World Network Properties')
plt.legend(loc='upper right')
plt.grid(True)

# Show plot
plt.show()
