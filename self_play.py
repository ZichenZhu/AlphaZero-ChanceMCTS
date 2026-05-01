"""
Self-play: generate training data by having the agent play against itself.

Each game produces a list of (observation, mcts_policy, outcome) tuples.
The agent uses MCTS to decide each move, and the final game result is
used as the training target for the value head.
"""

import os
import concurrent.futures
import numpy as np
import pyspiel
from tqdm import tqdm
from utils import encode_state, get_legal_actions_mask, apply_temperature
from mcts import MCTS, ChanceAwareMCTS
import torch
import multiprocessing as mp


def play_one_game(game, network, mcts_class, config, device="cpu"):
    """
    Play a single self-play game and collect training data.
    
    Args:
        game: OpenSpiel game object
        network: neural network for MCTS guidance
        mcts_class: MCTS or ChanceAwareMCTS
        config: hyperparameter config
        device: torch device
    
    Returns:
        training_examples: list of (observation, policy, value_target) tuples
    """

    with torch.no_grad():
        state = game.new_initial_state()
        mcts_engine = mcts_class(game, network, config, device)
    
    # Collect trajectory: (observation, mcts_policy, current_player)
    trajectory = []
    move_number = 0
    max_length = getattr(config, "max_game_length", 1000)
    num_actions = game.num_distinct_actions()
    
    while not state.is_terminal() and move_number < max_length:
        # Handle chance nodes (dice rolls in Backgammon)
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
            continue
        
        current_player = state.current_player()
        
        # Get observation for current player
        observation = encode_state(state, game)
        
        # Run MCTS to get improved policy
        mcts_policy = mcts_engine.search(state, add_noise=True)

        if hasattr(mcts_engine, 'cache'):
            mcts_engine.cache.clear()
        
        # Apply temperature for action selection
        if move_number < config.temp_threshold:
            temp = config.temp_init
        else:
            temp = config.temp_final
        
        action_probs = apply_temperature(mcts_policy, temp)
        
        # Store training example
        trajectory.append((observation, mcts_policy, current_player))
        
        # Select and apply action
        legal_actions = state.legal_actions()
        # Mask to legal actions and renormalize
        legal_probs = np.zeros(num_actions)
        for a in legal_actions:
            legal_probs[a] = action_probs[a]
        total = legal_probs.sum()
        if total > 0:
            legal_probs /= total
        else:
            # Fallback: uniform over legal actions
            for a in legal_actions:
                legal_probs[a] = 1.0 / len(legal_actions)
        
        action = np.random.choice(num_actions, p=legal_probs)
        state.apply_action(action)
        move_number += 1
    
    # Game is over — get the outcome
    if state.is_terminal():
        returns = state.returns()
    else:
        # Game exceeded max length, treat as draw
        returns = [0.0] * game.num_players()
    
    # Convert trajectory to training examples
    # value_target = game result from the perspective of the player who was moving
    training_examples = []
    for observation, policy, player in trajectory:
        value_target = returns[player]
        training_examples.append((observation, policy, value_target))

    return training_examples


def run_self_play(game, network, mcts_class, config, device="cpu"):
    """
    Run multiple self-play games and collect all training data using multiprocessing.
    
    Args:
        game: OpenSpiel game object
        network: neural network
        mcts_class: MCTS or ChanceAwareMCTS
        config: hyperparameter config
        device: torch device
    
    Returns:
        all_examples: list of (observation, policy, value_target) tuples
    """
    all_examples = []
    network.eval()
    
    # Determine number of parallel workers. Defaults to CPU count if not in config.
    num_workers = getattr(config, "num_workers", os.cpu_count() or 4)
    
    # Safely move network to CPU before pickling for multiprocessing to prevent CUDA crashes
    original_device = next(network.parameters()).device if list(network.parameters()) else device
    network.to("cpu")
    worker_device = "cpu" 
    
    futures = []
    
    # Spawn parallel processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(config.num_self_play_games):
            futures.append(
                executor.submit(
                    play_one_game, 
                    game, 
                    network, 
                    mcts_class, 
                    config, 
                    worker_device
                )
            )
        
        # tqdm tracks completion as parallel workers finish their games
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=config.num_self_play_games, 
                           desc="Self-play (Parallel)"):
            try:
                examples = future.result()
                all_examples.extend(examples)
            except Exception as e:
                print(f"Error in self-play worker: {e}")
                
    # Restore network to the original device (e.g., GPU) for the training phase
    network.to(original_device)
    
    print(f"  Generated {len(all_examples)} training examples "
          f"from {config.num_self_play_games} games")
    
    return all_examples