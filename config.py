"""
Hyperparameter configuration for AlphaZero on Connect Four and Backgammon.
All tunable values live here so nothing is hardcoded elsewhere.
"""


class ConnectFourConfig:
    """Config for the Connect Four reproduction (Phase 1)."""
    # Game
    game_name = "connect_four"

    # Neural network
    observation_shape = (3, 6, 7)  # 3 planes (player0, player1, empty) x 6 rows x 7 cols
    action_size = 7                # 7 columns
    hidden_size = 128              # wider network for stronger play
    num_hidden_layers = 3          # deeper network

    # MCTS
    num_simulations = 30          # more simulations = better policy targets
    c_puct = 1.5                   # exploration constant in PUCT formula
    dirichlet_alpha = 0.8          # noise alpha for root exploration
    dirichlet_epsilon = 0.25       # fraction of noise mixed into root prior

    # Temperature for action selection
    temp_threshold = 15            # move number after which temperature drops
    temp_init = 1.0                # temperature for early moves (exploration)
    temp_final = 0.1               # temperature for later moves (exploitation)

    # Self-play
    num_self_play_games = 100      # more games = more diverse training data
    max_replay_buffer = 40000      # rotate out old random data

    # Training
    num_iterations = 10            # total self-play -> train cycles
    epochs_per_iteration = 10      # training epochs per iteration
    batch_size = 256
    learning_rate = 0.001
    weight_decay = 1e-4
    value_loss_weight = 1.0        # weight for value loss vs policy loss

    # Evaluation
    num_eval_games = 100            # more games = less noisy win rate
    eval_every = 1                 # evaluate more frequently

    # Seeds for reproducibility
    seeds = [42, 123, 456]


class BackgammonConfig:
    """Config for the Backgammon adaptation (Phase 2)."""
    # Game
    game_name = "backgammon"

    # Neural network - Backgammon has a much larger state/action space
    observation_shape = None       # will be set dynamically from the game
    action_size = None             # will be set dynamically from the game
    hidden_size = 256
    num_hidden_layers = 4

    # MCTS
    num_simulations = 200          # balance between quality and speed
    c_puct = 1.5
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.25

    # Chance node handling
    chance_node_method = "sampling"  # "expectation" or "sampling"
    num_chance_samples = 6              # if using sampling, how many dice outcomes to sample

    # Temperature
    temp_threshold = 20
    temp_init = 1.0
    temp_final = 0.1

    # Self-play
    num_self_play_games = 50       # fewer than CF since BG games are slower
    max_replay_buffer = 50000
    max_game_length = 300          # terminate very long games

    # Training
    num_iterations = 25
    epochs_per_iteration = 10
    batch_size = 256
    learning_rate = 0.001
    weight_decay = 1e-4
    value_loss_weight = 1.0

    # Evaluation
    num_eval_games = 40
    eval_every = 5

    # Seeds
    seeds = [42, 123, 456]