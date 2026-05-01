"""
Quick test: verify OpenSpiel is installed and both games work.
Run this first: python test_env.py
"""

import pyspiel
import numpy as np
from utils import encode_state, get_legal_actions_mask, print_game_info


def test_connect_four():
    print("=" * 50)
    print("Testing Connect Four")
    print("=" * 50)
    
    game = pyspiel.load_game("connect_four")
    print_game_info(game)
    
    state = game.new_initial_state()
    print("Initial state:")
    print(state)
    
    # Check observation tensor
    obs = encode_state(state, game)
    print(f"Observation shape: {obs.shape}")
    
    # Check legal actions
    legal = state.legal_actions()
    print(f"Legal actions: {legal}")
    
    mask = get_legal_actions_mask(state, game.num_distinct_actions())
    print(f"Legal mask: {mask}")
    
    # Play a random game
    move_count = 0
    while not state.is_terminal():
        action = np.random.choice(state.legal_actions())
        state.apply_action(action)
        move_count += 1
    
    print(f"\nRandom game finished in {move_count} moves")
    print(f"Returns: {state.returns()}")
    print(f"Final state:")
    print(state)
    print("Connect Four: OK\n")


def test_backgammon():
    print("=" * 50)
    print("Testing Backgammon")
    print("=" * 50)
    
    game = pyspiel.load_game("backgammon")
    print_game_info(game)
    
    state = game.new_initial_state()
    
    # Backgammon starts with a chance node (dice roll)
    print(f"Is initial state a chance node? {state.is_chance_node()}")
    
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        print(f"Number of chance outcomes: {len(outcomes)}")
        print(f"First few outcomes (action, prob):")
        for action, prob in outcomes[:5]:
            print(f"  Action {action}: probability {prob:.4f}")
        
        # Apply a random chance outcome
        actions, probs = zip(*outcomes)
        action = np.random.choice(actions, p=probs)
        state.apply_action(action)
        print(f"\nAfter dice roll:")
    
    # Now it should be a player node
    print(f"Is chance node? {state.is_chance_node()}")
    print(f"Current player: {state.current_player()}")
    print(f"Legal actions: {state.legal_actions()[:10]}...")  # first 10
    print(f"Number of legal actions: {len(state.legal_actions())}")
    
    # Check observation
    obs = encode_state(state, game)
    print(f"Observation shape: {obs.shape}")
    
    # Play a short random game (Backgammon can be long)
    move_count = 0
    max_moves = 200
    while not state.is_terminal() and move_count < max_moves:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
        else:
            action = np.random.choice(state.legal_actions())
        state.apply_action(action)
        move_count += 1
    
    if state.is_terminal():
        print(f"\nRandom game finished in {move_count} moves")
        print(f"Returns: {state.returns()}")
    else:
        print(f"\nGame still going after {max_moves} moves (normal for Backgammon)")
    
    print("Backgammon: OK\n")


def test_model():
    print("=" * 50)
    print("Testing Neural Network")
    print("=" * 50)
    
    from model import create_network
    from config import ConnectFourConfig, BackgammonConfig
    
    # Test Connect Four model
    cf_config = ConnectFourConfig()
    cf_net = create_network(cf_config)
    print(f"Connect Four network parameters: {sum(p.numel() for p in cf_net.parameters()):,}")
    
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    obs = encode_state(state, game)
    mask = get_legal_actions_mask(state, game.num_distinct_actions())
    policy, value = cf_net.predict(obs, mask)
    print(f"  Policy shape: {policy.shape}, sums to: {policy.sum():.4f}")
    print(f"  Value: {value:.4f}")
    print(f"  Policy (legal actions only): {policy[policy > 0.01]}")
    
    # Test Backgammon model
    bg_config = BackgammonConfig()
    bg_net = create_network(bg_config)
    print(f"\nBackgammon network parameters: {sum(p.numel() for p in bg_net.parameters()):,}")
    print(f"  Observation shape: {bg_config.observation_shape}")
    print(f"  Action size: {bg_config.action_size}")
    
    print("\nNeural Network: OK\n")


if __name__ == "__main__":
    test_connect_four()
    test_backgammon()
    test_model()
    print("All tests passed!")