"""
Utility functions: state encoding, logging, plotting, and shared helpers.
"""

import os
import json
import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from datetime import datetime


def encode_state(state, game):
    """
    Convert an OpenSpiel state into a numpy observation tensor.
    Works for any game that provides observation tensors.
    
    Returns: numpy array of shape matching the game's observation tensor spec.
    """
    # Get the current player
    current_player = state.current_player()
    
    # Get observation tensor as flat array
    obs_tensor = np.array(state.observation_tensor(current_player), dtype=np.float32)
    
    # Reshape based on game's tensor shape
    tensor_shape = game.observation_tensor_shape()
    obs_tensor = obs_tensor.reshape(tensor_shape)
    
    return obs_tensor


def get_legal_actions_mask(state, num_actions):
    """
    Create a binary mask of legal actions.
    1 = legal, 0 = illegal.
    """
    mask = np.zeros(num_actions, dtype=np.float32)
    for action in state.legal_actions():
        mask[action] = 1.0
    return mask


def apply_temperature(visit_counts, temperature):
    """
    Convert MCTS visit counts to a probability distribution using temperature.
    
    temperature = 1.0: proportional to visit counts (more exploration)
    temperature -> 0: argmax (more exploitation)
    """
    if temperature < 0.01:
        # Nearly deterministic: pick the most visited action
        probs = np.zeros_like(visit_counts, dtype=np.float64)
        probs[np.argmax(visit_counts)] = 1.0
        return probs
    
    # Apply temperature
    counts = np.array(visit_counts, dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    total = counts.sum()
    if total == 0:
        # Uniform over non-zero counts
        nonzero = (np.array(visit_counts) > 0).astype(np.float64)
        return nonzero / nonzero.sum()
    return counts / total


def setup_results_dir(game_name, seed, mcts_class):
    """
    Create a clean, predictable results directory for a specific run 
    based on the game, algorithm mode, and seed.
    """
    # Determine the mode based on the MCTS class passed in
    if "ChanceAware" in mcts_class.__name__:
        mode = "adapted"
    else:
        mode = "unmodified"
        
    # Create a predictable path: results/backgammon_adapted_seed_42
    dir_name = f"{game_name}_{mode}_seed_{seed}"
    results_dir = os.path.join("results", dir_name)
    
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_training_history(history, filepath):
    """Save training metrics to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(filepath):
    """Load training metrics from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_training_curves(histories, labels, title, ylabel, filepath):
    """
    Plot training curves from multiple seeds on the same axes.
    
    histories: list of lists of values (one per seed)
    labels: list of labels for each seed
    title: plot title
    ylabel: y-axis label
    filepath: where to save the plot
    """
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label, linewidth=1.5)
    
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filepath}")


def plot_comparison(data_dict, title, ylabel, filepath):
    """
    Bar chart comparing different methods.
    
    data_dict: {method_name: (mean, std)} 
    """
    methods = list(data_dict.keys())
    means = [data_dict[m][0] for m in methods]
    stds = [data_dict[m][1] for m in methods]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, means, yerr=stds, capsize=5, 
                   color=["#4A90D9", "#D94A4A", "#4AD94A"][:len(methods)],
                   edgecolor="black", linewidth=0.5)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filepath}")


def print_game_info(game):
    """Print useful information about a game."""
    print(f"Game: {game.get_type().long_name}")
    print(f"  Players: {game.num_players()}")
    print(f"  Observation tensor shape: {game.observation_tensor_shape()}")
    print(f"  Num distinct actions: {game.num_distinct_actions()}")
    print(f"  Max game length: {game.max_game_length()}")
    
    game_type = game.get_type()
    print(f"  Chance mode: {game_type.chance_mode}")
    print(f"  Information: {game_type.information}")
    print(f"  Dynamics: {game_type.dynamics}")
    print()