from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import torch

def visualize_har_explanation(model, sample_input, target_label):
    """
    sample_input: tensor (1, 100, 3)
    """
    model.eval()
    ig = IntegratedGradients(model)
    
    # Identify which of the 100 timesteps were most important
    attributions, delta = ig.attribute(sample_input, target=target_label, return_convergence_delta=True)
    
    # Sum over X, Y, Z axes to get single importance score per timestep
    importance = attributions.squeeze().cpu().detach().numpy().sum(axis=-1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(importance, color='blue')
    plt.fill_between(range(100), importance, alpha=0.3)
    plt.title(f"XAI: Timestep Importance for Class {target_label}")
    plt.xlabel("Timestep (1-100)")
    plt.ylabel("Attribution Score")
    plt.show()