"""
Run the complete experiment pipeline for the final report.

This script runs ALL required experiments in order:
  1. Connect Four reproduction (3 seeds)
  2. Backgammon with standard MCTS (shows failure - 3 seeds)
  3. Backgammon with chance-aware MCTS (adaptation - 3 seeds)
  4. Generates all comparison plots

Usage:
  python run_experiments.py              # Full pipeline
  python run_experiments.py --quick      # Quick test with reduced settings
  python run_experiments.py --phase 1    # Only Connect Four
  python run_experiments.py --phase 2    # Only Backgammon (both variants)
"""

import argparse
import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

import pyspiel
from config import ConnectFourConfig, BackgammonConfig
from model import create_network
from mcts import MCTS, ChanceAwareMCTS
from train import run_training
from evaluate import evaluate_against_random, evaluate_head_to_head
from utils import plot_training_curves, plot_comparison, save_training_history
import multiprocessing as mp


def run_connect_four_reproduction(device="cpu", quick=False):
    """Phase 1: Reproduce AlphaZero on Connect Four."""
    print("\n" + "=" * 70)
    print("PHASE 1: Connect Four Reproduction")
    print("=" * 70)
    
    config = ConnectFourConfig()
    if quick:
        config.num_iterations = 5
        config.num_self_play_games = 10
        config.num_eval_games = 10
        config.eval_every = 2
        config.num_simulations = 25
    
    all_histories = []
    for seed in config.seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        history, results_dir = run_training(config, MCTS, current_seed=seed, device=device)
        all_histories.append(history)
    
    # Save aggregate results
    os.makedirs("results", exist_ok=True)
    
    # Plot win rates
    if all_histories[0]["win_rates"]:
        plot_training_curves(
            [h["win_rates"] for h in all_histories],
            [f"Seed {s}" for s in config.seeds],
            "Connect Four: Win Rate vs Random Player",
            "Win Rate",
            "results/connect_four_win_rates.png"
        )
    
    # Plot losses
    if all_histories[0]["policy_losses"]:
        plot_training_curves(
            [h["policy_losses"] for h in all_histories],
            [f"Seed {s}" for s in config.seeds],
            "Connect Four: Policy Loss",
            "Cross-Entropy Loss",
            "results/connect_four_policy_loss.png"
        )
        plot_training_curves(
            [h["value_losses"] for h in all_histories],
            [f"Seed {s}" for s in config.seeds],
            "Connect Four: Value Loss",
            "MSE Loss",
            "results/connect_four_value_loss.png"
        )
    
    with open("results/connect_four_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)
    
    return all_histories


def run_backgammon_unmodified(device="cpu", quick=False):
    """Phase 2a: Run standard MCTS on Backgammon (shows failure)."""
    print("\n" + "=" * 70)
    print("PHASE 2a: Backgammon with STANDARD MCTS (should fail/plateau)")
    print("=" * 70)
    
    config = BackgammonConfig()
    if quick:
        config.num_iterations = 5
        config.num_self_play_games = 5
        config.num_eval_games = 10
        config.eval_every = 2
        config.num_simulations = 25
    
    all_histories = []
    for seed in config.seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        history, results_dir = run_training(config, MCTS, current_seed=seed, device=device)
        all_histories.append(history)
    
    os.makedirs("results", exist_ok=True)
    with open("results/backgammon_unmodified_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)
    
    return all_histories


def run_backgammon_adapted(device="cpu", quick=False):
    """Phase 2b: Run chance-aware MCTS on Backgammon (adaptation)."""
    print("\n" + "=" * 70)
    print("PHASE 2b: Backgammon with CHANCE-AWARE MCTS (adaptation)")
    print("=" * 70)
    
    config = BackgammonConfig()
    if quick:
        config.num_iterations = 5
        config.num_self_play_games = 5
        config.num_eval_games = 10
        config.eval_every = 2
        config.num_simulations = 25
    
    all_histories = []
    for seed in config.seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        history, results_dir = run_training(config, ChanceAwareMCTS, current_seed=seed, device=device)
        all_histories.append(history)
    
    os.makedirs("results", exist_ok=True)
    with open("results/backgammon_adapted_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)
    
    return all_histories


def generate_comparison_plots(cf_histories, bg_unmod_histories, bg_adapted_histories):
    """Generate all comparison plots needed for the report."""
    print("\n" + "=" * 70)
    print("Generating comparison plots")
    print("=" * 70)
    
    os.makedirs("results", exist_ok=True)
    
    # --- Plot 1: Connect Four reproduction ---
    
    # --- Plot 2: Backgammon unmodified vs adapted win rates ---
    if bg_unmod_histories and bg_adapted_histories:
        plt.figure(figsize=(10, 6))
        
        # Plot unmodified
        for i, h in enumerate(bg_unmod_histories):
            if h["win_rates"]:
                label = "Standard MCTS" if i == 0 else None
                plt.plot(h["eval_iterations"], h["win_rates"],
                        "r--", alpha=0.5, label=label)
        
        # Plot adapted
        for i, h in enumerate(bg_adapted_histories):
            if h["win_rates"]:
                label = "Chance-Aware MCTS" if i == 0 else None
                plt.plot(h["eval_iterations"], h["win_rates"],
                        "b-", alpha=0.7, label=label)
        
        plt.xlabel("Training Iteration")
        plt.ylabel("Win Rate vs Random")
        plt.title("Backgammon: Standard vs Chance-Aware MCTS")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/backgammon_comparison.png", dpi=150)
        plt.close()
        print("Saved: results/backgammon_comparison.png")
    
    # --- Plot 3: Ablation bar chart ---
    # Compute final win rates for each method
    methods = {}
    
    if cf_histories:
        final_wr = [h["win_rates"][-1] for h in cf_histories if h["win_rates"]]
        if final_wr:
            methods["Connect Four\n(Standard MCTS)"] = (np.mean(final_wr), np.std(final_wr))
    
    if bg_unmod_histories:
        final_wr = [h["win_rates"][-1] for h in bg_unmod_histories if h["win_rates"]]
        if final_wr:
            methods["Backgammon\n(Standard MCTS)"] = (np.mean(final_wr), np.std(final_wr))
    
    if bg_adapted_histories:
        final_wr = [h["win_rates"][-1] for h in bg_adapted_histories if h["win_rates"]]
        if final_wr:
            methods["Backgammon\n(Chance-Aware)"] = (np.mean(final_wr), np.std(final_wr))
    
    if methods:
        plot_comparison(
            methods,
            "Final Win Rate vs Random (3 seeds)",
            "Win Rate",
            "results/ablation_comparison.png"
        )
    
    print("\nAll plots saved to results/")


def run_head_to_head_showdown(device="cpu", num_games=20):
    """
    Wrapper function to pit the Standard MCTS model against the Chance-Aware MCTS model.
    """
    print("\n" + "=" * 70)
    print("FINAL SHOWDOWN: Standard MCTS vs Chance-Aware MCTS")
    print("=" * 70)

    # 1. Load the configuration and game environment
    config = BackgammonConfig()
    game = pyspiel.load_game(config.game_name)

    # 2. Initialize two separate raw neural networks
    network_unmod = create_network(config, device)
    network_adapted = create_network(config, device)

    # 3. Define paths to your saved best models
    # NOTE: Adjust these folder names to exactly match what your setup_results_dir creates!
    seed = config.seeds[0]
    unmod_model_path = f"results/backgammon_unmodified_seed_{seed}/best_model.pth"
    adapted_model_path = f"results/backgammon_adapted_seed_{seed}/best_model.pth"

    # 4. Safely load the weights into the networks
    if os.path.exists(unmod_model_path) and os.path.exists(adapted_model_path):
        # map_location ensures it loads correctly whether you are on CPU, CUDA, or MPS
        network_unmod.load_state_dict(torch.load(unmod_model_path, map_location=device))
        network_adapted.load_state_dict(torch.load(adapted_model_path, map_location=device))
        print("Successfully loaded both best models from the results folder.")
    else:
        print(f"Error: Could not find model files.")
        print(f"Checked: {unmod_model_path}")
        print(f"Checked: {adapted_model_path}")
        print("Please check your folder structure and update the paths in this function.")
        return

    # 5. Run the evaluation
    print(f"Playing {num_games} games...")
    print("Player A: Standard MCTS | Player B: Chance-Aware MCTS")
    
    win_rate_a = evaluate_head_to_head(
        game=game,
        network_a=network_unmod,
        network_b=network_adapted,
        mcts_class_a=MCTS,
        mcts_class_b=ChanceAwareMCTS,
        config=config,
        num_games=num_games,
        device=device
    )

    # Print final summary
    print(f"\nFinal Result Summary:")
    print(f"Standard MCTS Win Rate: {win_rate_a:.1%}")
    print(f"Chance-Aware MCTS Win Rate: {1.0 - win_rate_a:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced settings")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="0=all, 1=Connect Four only, 2=Backgammon only")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"
    
    cf_histories = None
    bg_unmod = None
    bg_adapted = None
    
    if args.phase in [0, 1]:
        cf_histories = run_connect_four_reproduction(device, args.quick)
    
    if args.phase in [0, 2]:
        bg_unmod = run_backgammon_unmodified(device, args.quick)
        bg_adapted = run_backgammon_adapted(device, args.quick)
    
    # Try to load missing histories from disk
    if cf_histories is None and os.path.exists("results/connect_four_histories.json"):
        with open("results/connect_four_histories.json") as f:
            cf_histories = json.load(f)
    if bg_unmod is None and os.path.exists("results/backgammon_unmodified_histories.json"):
        with open("results/backgammon_unmodified_histories.json") as f:
            bg_unmod = json.load(f)
    if bg_adapted is None and os.path.exists("results/backgammon_adapted_histories.json"):
        with open("results/backgammon_adapted_histories.json") as f:
            bg_adapted = json.load(f)
    
    generate_comparison_plots(cf_histories, bg_unmod, bg_adapted)
    
    if args.phase in [0, 2]:
        run_head_to_head_showdown(device=device, num_games=20)
        
    print("\n" + "=" * 70)
    print("DONE! Check the results/ folder for plots and checkpoints.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()