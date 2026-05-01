"""
Evaluation: pit the trained AlphaZero agent against baselines.

Baselines:
  - Random player: picks uniformly from legal actions
  - Vanilla MCTS: MCTS with random rollouts (no neural network)
  
Reports win rate, draw rate, and loss rate.
"""

import numpy as np
import pyspiel
from tqdm import tqdm
from model import create_network
from utils import encode_state, get_legal_actions_mask
from mcts import MCTS, ChanceAwareMCTS
import concurrent.futures


def random_action(state):
    """Pick a random legal action."""
    legal = state.legal_actions()
    return np.random.choice(legal)


def mcts_action(state, game, network, mcts_class, config, device="cpu"):
    """
    Use MCTS to pick an action (exploitation mode: low temperature).
    """
    engine = mcts_class(game, network, config, device)
    policy = engine.search(state, add_noise=False)
    
    # Pick the action with highest visit count (greedy)
    legal = state.legal_actions()
    best_action = max(legal, key=lambda a: policy[a])
    return best_action


def play_one_eval_random(game, network_weights, mcts_class, config, device_str, agent_player):
    """Helper function for parallel evaluation against random player."""
    # Rebuild network to avoid memory locking in parallel processes
    worker_network = create_network(config, device_str) # Use your network init function
    worker_network.load_state_dict(network_weights)
    worker_network.to(device_str)
    worker_network.eval()
    
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
            continue
        
        current_player = state.current_player()
        
        if current_player == agent_player:
            # Using the MCTS engine for the agent
            action = mcts_action(state, game, worker_network, mcts_class, config, device_str)
        else:
            action = random_action(state)
            
        state.apply_action(action)
        
    returns = state.returns()
    return returns[agent_player]

def evaluate_against_random(game, network, mcts_class, config, num_games=40, device="cpu"):
    """Parallelized evaluation against a random player."""
    wins = 0
    draws = 0
    losses = 0
    
    # Safely move to CPU for weight extraction
    original_device = next(network.parameters()).device if list(network.parameters()) else device
    network.to("cpu")
    network_weights = network.state_dict()
    
    half = num_games // 2
    num_workers = getattr(config, "num_workers", 4)
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for game_idx in range(num_games):
            agent_player = 0 if game_idx < half else 1
            futures.append(
                executor.submit(
                    play_one_eval_random, game, network_weights, mcts_class, config, "cpu", agent_player
                )
            )
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_games, desc="Eval vs Random"):
            agent_return = future.result()
            if agent_return > 0:
                wins += 1
            elif agent_return == 0:
                draws += 1
            else:
                losses += 1
                
    network.to(original_device)
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    results = {
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": win_rate, "draw_rate": draw_rate, "loss_rate": loss_rate,
        "num_games": num_games,
    }
    
    print(f"  vs Random: W={wins} D={draws} L={losses} (win rate: {win_rate:.1%})")
    return win_rate, results


def play_one_head_to_head(game, weights_a, weights_b, mcts_class_a, mcts_class_b, config, device_str, a_player):
    """Helper function for parallel head-to-head evaluation."""
    
    import torch
    torch.set_num_threads(1)
    
    with torch.no_grad():
        # Rebuild networks inside the worker to avoid memory locking
        net_a = create_network(config, device_str)
        net_a.load_state_dict(weights_a)
        net_a.eval()
        
        net_b = create_network(config, device_str)
        net_b.load_state_dict(weights_b)
        net_b.eval()
        
        state = game.new_initial_state()
        
        move_number = 0
        max_length = getattr(config, "max_game_length", 1000)
        
        while not state.is_terminal() and move_number < max_length:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
                continue
            
            current_player = state.current_player()
            
            if current_player == a_player:
                action = mcts_action(state, game, net_a, mcts_class_a, config, device_str)
            else:
                action = mcts_action(state, game, net_b, mcts_class_b, config, device_str)
                
            state.apply_action(action)
            move_number += 1
            
        # Get returns safely
        if state.is_terminal():
            returns = state.returns()
        else:
            # Treat incomplete games due to loop limits as a draw
            returns = [0.0] * game.num_players()
            
        return returns[a_player]


def evaluate_head_to_head(game, network_a, network_b, mcts_class_a, mcts_class_b, config, num_games=20, device="cpu"):
    """Parallelized head-to-head evaluation."""
    wins_a = 0
    draws = 0
    
    # Extract weights to pass to CPU workers safely
    network_a.to("cpu")
    network_b.to("cpu")
    weights_a = network_a.state_dict()
    weights_b = network_b.state_dict()
    
    half = num_games // 2
    num_workers = getattr(config, "num_workers", 4)
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for game_idx in range(num_games):
            # Player A is player 0 for the first half, player 1 for the second half
            a_player = 0 if game_idx < half else 1
            
            futures.append(
                executor.submit(
                    play_one_head_to_head, 
                    game, weights_a, weights_b, 
                    mcts_class_a, mcts_class_b, 
                    config, "cpu", a_player
                )
            )
            
        # Collect results as they finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_games, desc="Head-to-head"):
            agent_return = future.result()
            if agent_return > 0:
                wins_a += 1
            elif agent_return == 0:
                draws += 1
                
    # Restore original devices if needed (assuming main script handles it)
    win_rate_a = wins_a / num_games
    print(f"  Head-to-head: A wins {wins_a}/{num_games} ({win_rate_a:.1%})")
    
    return win_rate_a