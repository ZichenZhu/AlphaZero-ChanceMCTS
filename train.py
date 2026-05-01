"""
Main training loop for AlphaZero.

Orchestrates the cycle:
  1. Self-play: generate games using MCTS + neural network
  2. Train: update the neural network on the collected data
  3. Evaluate: test against a random player to track progress

Usage:
  # Connect Four reproduction (Phase 1)
  python train.py --game connect_four --seed 42

  # Backgammon with chance-aware MCTS (Phase 2 adaptation)
  python train.py --game backgammon --seed 42

  # Backgammon with STANDARD MCTS (to show it fails - ablation)
  python train.py --game backgammon --seed 42 --no-chance-aware
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import pyspiel
from config import ConnectFourConfig, BackgammonConfig
from model import create_network
from mcts import MCTS, ChanceAwareMCTS
from self_play import run_self_play
from evaluate import evaluate_against_random
from utils import (
    setup_results_dir,
    save_training_history,
    plot_training_curves,
)
import multiprocessing as mp


def train_network(network, replay_buffer, config, optimizer, device="cpu"):
    """
    Train the neural network on collected self-play data.
    
    Loss = policy_loss (cross-entropy) + value_loss_weight * value_loss (MSE)
    
    Args:
        network: the AlphaZero neural network
        replay_buffer: list of (observation, policy, value_target) tuples
        config: hyperparameter config
        optimizer: torch optimizer (persisted across iterations)
        device: torch device
    
    Returns:
        avg_policy_loss, avg_value_loss over this training session
    """
    network.train()
    
    # Convert data to tensors
    observations = np.array([ex[0] for ex in replay_buffer], dtype=np.float32)
    target_policies = np.array([ex[1] for ex in replay_buffer], dtype=np.float32)
    target_values = np.array([ex[2] for ex in replay_buffer], dtype=np.float32)
    
    # Flatten observations
    observations = observations.reshape(len(observations), -1)
    
    dataset_size = len(observations)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    for epoch in range(config.epochs_per_iteration):
        # Shuffle
        indices = np.random.permutation(dataset_size)
        
        for start in range(0, dataset_size, config.batch_size):
            end = min(start + config.batch_size, dataset_size)
            batch_idx = indices[start:end]
            
            obs_batch = torch.FloatTensor(observations[batch_idx]).to(device)
            policy_batch = torch.FloatTensor(target_policies[batch_idx]).to(device)
            value_batch = torch.FloatTensor(target_values[batch_idx]).unsqueeze(1).to(device)
            
            # Forward pass
            policy_logits, value_pred = network(obs_batch)
            
            # Policy loss: cross-entropy between MCTS policy and network policy
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(policy_batch * log_probs, dim=1).mean()
            
            # Value loss: MSE between game outcome and predicted value
            value_loss = nn.MSELoss()(value_pred, value_batch)
            
            # Total loss
            loss = policy_loss + config.value_loss_weight * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_policy_loss = total_policy_loss / max(num_batches, 1)
    avg_value_loss = total_value_loss / max(num_batches, 1)
    
    return avg_policy_loss, avg_value_loss


def run_training(config, mcts_class, current_seed, device="cpu"):
    """
    Full AlphaZero training loop for one specific seed.
    """
    game = pyspiel.load_game(config.game_name)
    network = create_network(config, device)
    
    # Create optimizer ONCE
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Replay buffer
    replay_buffer = deque(maxlen=config.max_replay_buffer)
    
    history = {
        "policy_losses": [],
        "value_losses": [],
        "win_rates": [],
        "eval_iterations": [],
    }
    
    results_dir = setup_results_dir(config.game_name, current_seed, mcts_class)
    
    best_win_rate = -1.0
    
    for iteration in range(1, config.num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{config.num_iterations} (Seed: {current_seed})")
        print(f"{'='*60}")
        
        # 1. Self-play
        print("Phase 1: Self-play...")
        examples = run_self_play(game, network, mcts_class, config, device)
        replay_buffer.extend(examples)
        print(f"  Replay buffer size: {len(replay_buffer)}")
        
        # 2. Train
        print("Phase 2: Training...")
        if len(replay_buffer) >= config.batch_size:
            policy_loss, value_loss = train_network(
                network, list(replay_buffer), config, optimizer, device
            )
            history["policy_losses"].append(policy_loss)
            history["value_losses"].append(value_loss)
            print(f"  Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
        
        # 3. Evaluate and Checkpoint
        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            print("Phase 3: Evaluation...")
            win_rate, _ = evaluate_against_random(
                game, network, mcts_class, config,
                num_games=config.num_eval_games, device=device
            )
            history["win_rates"].append(win_rate)
            history["eval_iterations"].append(iteration)
            
            if win_rate > best_win_rate:
                print(f"  New best win rate achieved: {win_rate:.1%} (Previous: {max(0, best_win_rate):.1%})")
                print("  Saving new best model...")
                best_win_rate = win_rate
                best_path = os.path.join(results_dir, "best_model.pth")
                torch.save(network.state_dict(), best_path)
            else:
                print(f"  Win rate {win_rate:.1%} did not beat the best {best_win_rate:.1%}. Model not saved.")
    
    final_path = os.path.join(results_dir, "final_model.pth")
    torch.save(network.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"Best model preserved at {os.path.join(results_dir, 'best_model.pth')} with a win rate of {best_win_rate:.1%}")
    
    history_path = os.path.join(results_dir, "training_history.json")
    save_training_history(history, history_path)
    
    return history, results_dir


def run_all_seeds(config, mcts_class, device="cpu"):
    """
    Run training across all seeds and produce aggregate plots.
    """
    all_histories = []
    
    for seed in config.seeds:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # PASS THE CURRENT SEED TO RUN_TRAINING
        history, results_dir = run_training(config, mcts_class, current_seed=seed, device=device)
        all_histories.append(history)
    
    # Plot aggregate results
    os.makedirs("results", exist_ok=True)
    
    # Distinguish mode for plot filenames
    if "ChanceAware" in mcts_class.__name__:
        mode = "adapted"
    else:
        mode = "unmodified"
    
    # Win rate curves
    if all_histories[0]["win_rates"]:
        plot_training_curves(
            [h["win_rates"] for h in all_histories],
            [f"Seed {s}" for s in config.seeds],
            f"Win Rate vs Random - {config.game_name} ({mode})",
            "Win Rate",
            f"results/{config.game_name}_{mode}_win_rates.png"
        )
    
    # Loss curves
    if all_histories[0]["policy_losses"]:
        plot_training_curves(
            [h["policy_losses"] for h in all_histories],
            [f"Seed {s}" for s in config.seeds],
            f"Policy Loss - {config.game_name} ({mode})",
            "Policy Loss",
            f"results/{config.game_name}_{mode}_policy_loss.png"
        )
    
    return all_histories


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument("--game", type=str, default="connect_four",
                        choices=["connect_four", "backgammon"],
                        help="Game to train on")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run (overrides config.seeds)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run all 3 seeds from config")
    parser.add_argument("--no-chance-aware", action="store_true",
                        help="Use standard MCTS even for Backgammon (ablation)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    args = parser.parse_args()
    
    # Select config
    if args.game == "connect_four":
        config = ConnectFourConfig()
        mcts_class = MCTS  # Connect Four is deterministic
    elif args.game == "backgammon":
        config = BackgammonConfig()
        if args.no_chance_aware:
            mcts_class = MCTS  # Ablation: standard MCTS on stochastic game
            print("WARNING: Using standard MCTS on Backgammon (ablation mode)")
        else:
            mcts_class = ChanceAwareMCTS  # Adapted MCTS
    
    # Override seed if specified
    if args.seed is not None:
        config.seeds = [args.seed]
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Training AlphaZero on {args.game}")
    print(f"MCTS class: {mcts_class.__name__}")
    print(f"Device: {device}")
    print(f"Seeds: {config.seeds}")
    
    if args.all_seeds or args.seed is None:
        run_all_seeds(config, mcts_class, device)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        run_training(config, mcts_class, current_seed=args.seed, device=device)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()