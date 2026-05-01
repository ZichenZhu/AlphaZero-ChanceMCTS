"""
Neural network model for AlphaZero.
Takes a board observation tensor and outputs:
  - policy: probability distribution over all actions
  - value: scalar estimate of position value from current player's perspective

Uses a simple MLP architecture for speed. Can be swapped for a ResNet 
if you have more compute budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNetwork(nn.Module):
    """
    MLP-based policy-value network for AlphaZero.
    
    Input: observation tensor (flattened)
    Output: (policy logits, value)
    
    Architecture:
        observation -> [hidden layers with ReLU + BatchNorm] -> policy head
                                                              -> value head
    """
    
    def __init__(self, observation_size, action_size, hidden_size=128, num_hidden_layers=3):
        """
        Args:
            observation_size: total number of elements in the observation tensor
            action_size: number of possible actions
            hidden_size: width of hidden layers
            num_hidden_layers: depth of the shared torso
        """
        super().__init__()
        
        self.observation_size = observation_size
        self.action_size = action_size
        
        # Shared torso
        layers = []
        in_size = observation_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.torso = nn.Sequential(*layers)
        
        # Policy head: outputs logits over actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        
        # Value head: outputs a scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor of shape (batch_size, observation_size)
        
        Returns:
            policy_logits: (batch_size, action_size) - raw logits (not softmaxed)
            value: (batch_size, 1) - position value in [-1, 1]
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        shared = self.torso(x)
        policy_logits = self.policy_head(shared)
        value = self.value_head(shared)
        
        return policy_logits, value
    
    def predict(self, observation, legal_actions_mask, device="cpu"):
        """
        Single-state prediction for MCTS.
        
        Args:
            observation: numpy array of the observation tensor
            legal_actions_mask: numpy array, 1 for legal actions, 0 for illegal
            device: torch device
        
        Returns:
            policy: numpy array, probability distribution over actions (masked to legal)
            value: float, position value estimate
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            if obs_tensor.dim() > 2:
                obs_tensor = obs_tensor.view(1, -1)
            
            policy_logits, value = self.forward(obs_tensor)
            
            # Mask illegal actions with large negative number before softmax
            mask_tensor = torch.FloatTensor(legal_actions_mask).to(device)
            masked_logits = policy_logits.squeeze(0)
            masked_logits = masked_logits - (1 - mask_tensor) * 1e9
            
            # Softmax to get probabilities
            policy = F.softmax(masked_logits, dim=0).cpu().numpy()
            value = value.item()
        
        return policy, value


def create_network(config, device="cpu"):
    """
    Factory function to create a network from a config object.
    Handles dynamic observation/action sizes for different games.
    """
    import pyspiel
    
    game = pyspiel.load_game(config.game_name)
    
    # Get observation size
    if config.observation_shape is not None:
        obs_size = 1
        for dim in config.observation_shape:
            obs_size *= dim
    else:
        obs_shape = game.observation_tensor_shape()
        obs_size = 1
        for dim in obs_shape:
            obs_size *= dim
        config.observation_shape = tuple(obs_shape)
    
    # Get action size
    if config.action_size is None:
        config.action_size = game.num_distinct_actions()
    
    action_size = config.action_size
    
    network = AlphaZeroNetwork(
        observation_size=obs_size,
        action_size=action_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    
    return network