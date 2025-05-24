
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(attn_weights, title="Transformer Attention"):
    """Visualize attention weights from Transformer layer"""
    if attn_weights is None:
        print("No attention weights to visualize.")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_weights[0].cpu().numpy(), cmap='viridis', annot=True)
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Token")
    plt.show()

def plot_uncertainty_distribution(mean_probs, std_probs):
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(mean_probs)), mean_probs, yerr=std_probs, fmt='o')
    plt.title("Prediction Uncertainty Distribution")
    plt.xlabel("Sample Index")
    plt.ylabel("Interaction Probability")
    plt.grid(True)
    plt.show()