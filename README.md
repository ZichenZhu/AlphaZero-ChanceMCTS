# AlphaZero on Connect Four and Backgammon

Reproducing and extending AlphaZero (Silver et al., 2017) using OpenSpiel.

Build a course-scale AlphaZero pipeline on Connect Four, then adapt it to Backgammon by modifying MCTS to handle chance nodes (dice rolls).

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify everything works
python test_env.py
```

## Project Structure

```
├── config.py            # All hyperparameters for both games
├── model.py             # Neural network (policy + value heads)
├── mcts.py              # Standard MCTS + chance-aware MCTS adaptation
├── self_play.py          # Self-play data generation
├── train.py             # Training loop (self-play → train → evaluate)
├── evaluate.py          # Evaluation against random/MCTS baselines
├── utils.py             # State encoding, plotting, helpers
├── run_experiments.py   # Runs all experiments and generates plots
├── test_env.py          # Environment verification script
└── results/             # Checkpoints, plots, training logs
```

## Running Experiments

### Quick test (verify everything works, ~10 min)
```bash
python run_experiments.py --quick
```
### Phase-Specific Runs
```bash
# Only run the Connect Four reproduction (Phase 1)
python run_experiments.py --phase 1

# Only run the Backgammon variants (Phase 2)
python run_experiments.py --phase 2
```

### Full pipeline (all experiments for the report)
```bash
python run_experiments.py --device cuda
```

### Individual runs
```bash
# Connect Four reproduction (single seed)
python train.py --game connect_four --seed 42

# Backgammon with chance-aware MCTS (adaptation)
python train.py --game backgammon --seed 42

# Backgammon with standard MCTS (ablation — shows failure)
python train.py --game backgammon --seed 42 --no-chance-aware

# All 3 seeds for one game
python train.py --game connect_four --all-seeds
```

### GPU support
```bash
python run_experiments.py --device cuda
```

## Key Algorithmic Details

### Standard MCTS (Connect Four)
- PUCT selection: UCB(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
- Neural network provides prior policy P and value estimate V
- Dirichlet noise at root for exploration

### Chance-Aware MCTS (Backgammon adaptation)
- **Core change**: When MCTS encounters a chance node (dice roll), it creates
  expectation nodes that expand all possible outcomes weighted by probability
- Configurable Strategies: The search supports both an expectation method (expanding all possible outcomes) and a sampling method (sampling a subset of dice rolls). These can be toggled via the chance_node_method variable in config.py.
- Chance nodes select children by their natural probability, NOT by PUCT
- Backup uses an absolute returns array [value_for_player0, value_for_player1]
  so each node receives the correct value from its parent's perspective,
  regardless of whose turn it is at the leaf
- This is the non-trivial algorithmic modification required by the project

## Expected Output

After running `run_experiments.py`, the `results/` folder contains:
- `connect_four_win_rates.png` — reproduction curve showing improvement
- `backgammon_comparison.png` — standard vs chance-aware MCTS
- `ablation_comparison.png` — bar chart comparing all three conditions
- Training histories as JSON files
- Model checkpoints

## References

- Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. arXiv:1712.01815
- OpenSpiel documentation: https://openspiel.readthedocs.io/
