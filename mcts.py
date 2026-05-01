"""
Monte Carlo Tree Search (MCTS) for AlphaZero.

Two variants:
  1. Standard MCTS — for deterministic games like Connect Four.
     On stochastic games (Backgammon ablation), it resolves chance nodes
     randomly and does NOT build tree structure past chance boundaries.
     After a player action leads to a chance node, it resolves the dice
     randomly and evaluates with the NN — making the search effectively
     one player-move deep. This is the WRONG approach for stochastic
     games and will perform poorly.
     
  2. Chance-aware MCTS — for stochastic games like Backgammon.
     Properly models chance nodes in the tree by creating branches for
     each possible dice outcome weighted by probability. This allows
     the search to look multiple moves ahead through chance events.

The key insight for Backgammon: after every player move, dice are rolled
(chance node) before the next player acts. Standard MCTS can't reuse
tree nodes past a chance boundary because the same action leads to
different game states depending on the dice. Chance-aware MCTS solves
this by explicitly branching over dice outcomes in the tree.
"""

import math
import numpy as np
import pyspiel
from utils import encode_state, get_legal_actions_mask


class MCTSNode:
    """A single node in the MCTS tree."""
    
    def __init__(self, prior, player):
        self.prior = prior
        self.player = player
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}
        self.is_expanded = False
        self.is_chance = False
    
    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def select_child(self, c_puct):
        """Select child with highest PUCT score."""
        best_score = -float("inf")
        best_action = None
        best_child = None
        
        sqrt_parent = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            ucb = child.value + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        
        return best_action, best_child


def _resolve_chance_nodes(state):
    """Resolve all consecutive chance nodes by sampling randomly."""
    while not state.is_terminal() and state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(np.random.choice(actions, p=probs))


class MCTS:
    """
    Standard MCTS for deterministic games (Connect Four).
    
    On stochastic games (Backgammon ablation): after a player action
    leads to a chance node, the dice are resolved randomly and the
    resulting state is evaluated directly by the NN.
    """
    
    def __init__(self, game, network, config, device="cpu"):
        self.game = game
        self.network = network
        self.config = config
        self.device = device
        self.num_actions = game.num_distinct_actions()
        self.cache = {}
    
    def search(self, state, add_noise=True):
        """Run MCTS and return visit count proportions over actions."""
        root = MCTSNode(prior=0, player=state.current_player())
        self._expand(root, state)
        
        if add_noise:
            self._add_dirichlet_noise(root)
        
        for _ in range(self.config.num_simulations):
            node = root
            search_state = state.clone()
            path = [node]
            
            # SELECT
            while node.is_expanded and node.children:
                action, node = node.select_child(self.config.c_puct)
                search_state.apply_action(action)
                path.append(node)
                
                if not search_state.is_terminal() and search_state.is_chance_node():
                    _resolve_chance_nodes(search_state)
                    break
            
            # EVALUATE
            if search_state.is_terminal():
                returns = search_state.returns()
            elif node.is_expanded:
                # Broke out of selection due to chance node. Evaluate leaf directly.
                obs = encode_state(search_state, self.game)
                mask = get_legal_actions_mask(search_state, self.num_actions)
                _, value = self.network.predict(obs, mask, self.device)
                
                # Build absolute returns array
                leaf_player = search_state.current_player()
                returns = [value if i == leaf_player else -value for i in range(2)]
            else:
                # Unexpanded leaf - expand it
                _resolve_chance_nodes(search_state)
                if search_state.is_terminal():
                    returns = search_state.returns()
                else:
                    value = self._expand(node, search_state)
                    leaf_player = search_state.current_player()
                    returns = [value if i == leaf_player else -value for i in range(2)]
            
            # BACKUP
            self._backup(path, returns)
        
        # Return visit count proportions
        action_probs = np.zeros(self.num_actions, dtype=np.float64)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        
        return action_probs
    
    def _expand(self, node, state):
        node.player = state.current_player()

        state_key = str(state)
        if state_key in self.cache:
            policy, value = self.cache[state_key]
        else:
            obs = encode_state(state, self.game)
            mask = get_legal_actions_mask(state, self.num_actions)
            policy, value = self.network.predict(obs, mask, self.device)
            self.cache[state_key] = (policy, value)
        
        for action in state.legal_actions():
            node.children[action] = MCTSNode(
                prior=policy[action],
                player=1 - state.current_player()
            )
        
        node.is_expanded = True
        return value
    
    def _backup(self, path, returns):
        """Backup values using the absolute perspective of the parent node."""
        for i in range(len(path)):
            node = path[i]
            node.visit_count += 1
            
            if i > 0:
                parent = path[i - 1]
                # Node stores value from the parent's perspective for UCB formula
                if parent.player >= 0:
                    node.total_value += returns[parent.player]
            else:
                # Root node
                p = node.player if node.player >= 0 else 0
                node.total_value += returns[p]
    
    def _add_dirichlet_noise(self, node):
        actions = list(node.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            node.children[action].prior = (
                (1 - eps) * node.children[action].prior + eps * noise[i]
            )


class ChanceAwareMCTS:
    """
    MCTS that properly handles chance nodes in the search tree.
    """
    
    def __init__(self, game, network, config, device="cpu"):
        self.game = game
        self.network = network
        self.config = config
        self.device = device
        self.num_actions = game.num_distinct_actions()
        self.chance_method = getattr(config, "chance_node_method", "expectation")
        self.num_chance_samples = getattr(config, "num_chance_samples", 6)
        self.cache = {}
    
    def search(self, state, add_noise=True):
        if state.is_chance_node():
            raise ValueError("Cannot start search from a chance node.")
        
        root = MCTSNode(prior=0, player=state.current_player())
        self._expand_player_node(root, state)
        
        if add_noise:
            self._add_dirichlet_noise(root)
        
        for _ in range(self.config.num_simulations):
            node = root
            search_state = state.clone()
            path = [node]
            
            # SELECT
            while node.is_expanded and node.children:
                if node.is_chance:
                    action = self._sample_chance_child(node)
                else:
                    action, _ = node.select_child(self.config.c_puct)
                
                search_state.apply_action(action)
                node = node.children[action]
                path.append(node)
            
            # EVALUATE
            if search_state.is_terminal():
                returns = search_state.returns()
            elif search_state.is_chance_node():
                self._expand_chance_node(node, search_state)
                action = self._sample_chance_child(node)
                search_state.apply_action(action)
                child = node.children[action]
                path.append(child)
                
                if search_state.is_terminal():
                    returns = search_state.returns()
                else:
                    value = self._expand_player_node(child, search_state)
                    leaf_player = search_state.current_player()
                    returns = [value if i == leaf_player else -value for i in range(2)]
            else:
                value = self._expand_player_node(node, search_state)
                leaf_player = search_state.current_player()
                returns = [value if i == leaf_player else -value for i in range(2)]
            
            # BACKUP
            self._backup(path, returns)
        
        # Return visit count proportions
        action_probs = np.zeros(self.num_actions, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        
        return action_probs
    
    def _expand_player_node(self, node, state):
        if state.is_terminal():
            node.is_expanded = True
            # Return value is no longer used directly by backup, handled by returns array
            return 0.0 
        
        node.player = state.current_player()
        node.is_chance = False

        state_key = str(state)
        if state_key in self.cache:
            policy, value = self.cache[state_key]
        else:
            obs = encode_state(state, self.game)
            mask = get_legal_actions_mask(state, self.num_actions)
            policy, value = self.network.predict(obs, mask, self.device)
            self.cache[state_key] = (policy, value)
        
        for action in state.legal_actions():
            node.children[action] = MCTSNode(prior=policy[action], player=-1)
        
        node.is_expanded = True
        return value
    
    def _expand_chance_node(self, node, state):
        node.is_chance = True
        node.is_expanded = True
        node.player = -1
        
        chance_outcomes = state.chance_outcomes()
        
        if self.chance_method == "expectation":
            for action, prob in chance_outcomes:
                node.children[action] = MCTSNode(prior=prob, player=-1)
        
        elif self.chance_method == "sampling":
            actions = [a for a, _ in chance_outcomes]
            probs = [p for _, p in chance_outcomes]
            n = min(self.num_chance_samples, len(actions))
            sampled = np.random.choice(len(actions), size=n, replace=False, p=probs)
            
            # Normalize the sampled subset so probabilities sum to 1
            sampled_probs = np.array([probs[idx] for idx in sampled])
            sampled_probs /= sampled_probs.sum()
            
            for i, idx in enumerate(sampled):
                node.children[actions[idx]] = MCTSNode(prior=sampled_probs[i], player=-1)
    
    def _sample_chance_child(self, node):
        actions = list(node.children.keys())
        probs = np.array([node.children[a].prior for a in actions])
        return actions[np.random.choice(len(actions), p=probs)]
    
    def _backup(self, path, returns):
        """Backup values using the absolute perspective of the parent node."""
        for i in range(len(path)):
            node = path[i]
            node.visit_count += 1
            
            if i > 0:
                parent = path[i - 1]
                if parent.player >= 0:
                    node.total_value += returns[parent.player]
                else:
                    # Parent is chance, doesn't use UCB. Give node absolute value.
                    p = node.player if node.player >= 0 else 0
                    node.total_value += returns[p]
            else:
                p = node.player if node.player >= 0 else 0
                node.total_value += returns[p]
    
    def _add_dirichlet_noise(self, node):
        actions = list(node.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            node.children[action].prior = (
                (1 - eps) * node.children[action].prior + eps * noise[i]
            )