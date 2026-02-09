from typing import Literal
import torch
import torch.nn as nn
from rsl_rl.modules.finite_scalar_quantization import FSQ
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

class Actor_Projection(nn.Module):
    """Multi-modal actor network with projection-based alignment.
    
    This network projects robot and human motion states to a shared latent space
    and decodes them into actions. The activate_signals parameter controls which
    modality branch is used during forward pass.
    
    Network Architecture:
        Robot Projection:
            - Input: state_goal (actor_sg_dim)
            - Hidden layers: robot_projection_hidden_dims
            - Output: latent vector (projection_hidden_dims)
        
        Human Projection:
            - Input: smplx_human_state (actor_sh_dim)
            - Hidden layers: human_projection_hidden_dims
            - Output: latent vector (projection_hidden_dims)
        
        Action Decoder:
            - Input: projected latent (projection_hidden_dims) + proprioceptive state (actor_sp_dim)
            - Hidden layers: action_hidden_dims
            - Output: action (num_actions)
    
    Signal flow:
        state_goal ──robot_projection── latent ──┐
                                                  ├──[concat with prop_state]──action_decoder── action
        proprioceptive_state ──normalizer────────┤
                                                  │
        smplx_state ──human_projection── latent ──┘
    """
    
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int, # robot goal state dimension
        actor_sh_dim: int, # smplx human state dimension
        # route parameters
        activate_signals: Literal["robot", "smplx"] = "robot",
        # Network Architecture
        robot_projection_hidden_dims: list[int] = [512, 256],
        human_projection_hidden_dims: list[int] = [512, 256],
        projection_hidden_dims: int = 64, 
        action_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.num_actions = num_actions
        self.projection_hidden_dims = projection_hidden_dims
        self.activate_signals = activate_signals
        
        activation_fn = resolve_nn_activation(activation)
        
        # Calculate proprioceptive state dimension
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim
        
        # Robot Projection Network: state_goal -> projection_hidden_dims
        robot_projection_layers = []
        robot_projection_layers.append(nn.Linear(actor_sg_dim, robot_projection_hidden_dims[0]))
        robot_projection_layers.append(activation_fn)
        for i in range(len(robot_projection_hidden_dims) - 1):
            robot_projection_layers.append(nn.Linear(robot_projection_hidden_dims[i], robot_projection_hidden_dims[i+1]))
            robot_projection_layers.append(activation_fn)
        robot_projection_layers.append(nn.Linear(robot_projection_hidden_dims[-1], projection_hidden_dims))
        self.robot_projection = nn.Sequential(*robot_projection_layers)
        
        # Human Projection Network: smplx_human_state -> projection_hidden_dims
        human_projection_layers = []
        human_projection_layers.append(nn.Linear(actor_sh_dim, human_projection_hidden_dims[0]))
        human_projection_layers.append(activation_fn)
        for i in range(len(human_projection_hidden_dims) - 1):
            human_projection_layers.append(nn.Linear(human_projection_hidden_dims[i], human_projection_hidden_dims[i+1]))
            human_projection_layers.append(activation_fn)
        human_projection_layers.append(nn.Linear(human_projection_hidden_dims[-1], projection_hidden_dims))
        self.human_projection = nn.Sequential(*human_projection_layers)
        
        # Proprioceptive State Normalizer
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # Action Decoder: projected_latent + proprioceptive_state -> actions
        action_decoder_input_dim = projection_hidden_dims + actor_sp_dim
        action_layers = []
        action_layers.append(nn.Linear(action_decoder_input_dim, action_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_hidden_dims[i], action_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)
    
    def forward(self, x):
        """Forward pass through the projection-based actor network.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               where first actor_sg_dim elements are robot goal state,
               next actor_sh_dim elements are human state,
               and remaining elements are proprioceptive state.
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        if self.activate_signals == "robot":
            actions = self.forward_robot(x)
        elif self.activate_signals == "smplx":
            actions = self.forward_smplx(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot' or 'smplx'.")
        
        return actions
    
    def forward_robot(self, x):
        """Forward pass using robot goal state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input into sg (robot state) and sp (proprioceptive state)
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Project robot state to latent space
        z_robot = self.robot_projection(x_sg)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_robot, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        """Forward pass using SMPLX human state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input into sh (human state) and sp (proprioceptive state)
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Project human state to latent space
        z_human = self.human_projection(x_sh)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_human, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_robot_exporter(self, robot_command, proprioceptive_state):
        """Forward pass for robot command and proprioceptive state.
        
        Args:
            robot_command: Input tensor of shape (batch_size, actor_sg_dim)
            proprioceptive_state: Input tensor of shape (batch_size, num_actor_obs - actor_sg_dim - actor_sh_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Project robot state to latent space
        z_robot = self.robot_projection(robot_command)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_robot, proprioceptive_state_normalized], dim=-1))
        
        return actions
    
    def forward_smplx_exporter(self, smplx_human_state, proprioceptive_state):
        """Forward pass for SMPLX human state and proprioceptive state.
        
        Args:
            smplx_human_state: Input tensor of shape (batch_size, actor_sh_dim)
            proprioceptive_state: Input tensor of shape (batch_size, num_actor_obs - actor_sg_dim - actor_sh_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Project human state to latent space
        z_human = self.human_projection(smplx_human_state)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_human, proprioceptive_state_normalized], dim=-1))
        
        return actions
    
    def project_robot(self, x):
        """Project robot goal state to shared latent space.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sg_dim)
        
        Returns:
            z_robot: Projected latent vector of shape (batch_size, projection_hidden_dims)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sg = x[:, :self.actor_sg_dim]
        else:
            x_sg = x
        
        return self.robot_projection(x_sg)
    
    def project_smplx(self, x):
        """Project SMPLX human state to shared latent space.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sh_dim)
        
        Returns:
            z_human: Projected latent vector of shape (batch_size, projection_hidden_dims)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        else:
            x_sh = x
        
        return self.human_projection(x_sh)
    
    def decode_action(self, z, x_sp):
        """Decode actions from projected latent and proprioceptive state.
        
        Args:
            z: Projected latent vector of shape (batch_size, projection_hidden_dims)
            x_sp: Proprioceptive state of shape (batch_size, num_actor_obs - actor_sg_dim - actor_sh_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        
        NOTE:
            z can be from either robot or human projection.
        """
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.action_decoder(torch.cat([z, x_sp_normalized], dim=-1))
    
    def train(self, mode: bool = True):
        """Override train() to ensure frozen modules stay in eval mode during finetune.
        
        The proprioceptive_normalizer is kept in training mode to continuously
        update its normalization statistics.
        """
        super().train(mode)
        # Ensure frozen modules remain in eval mode if mode=True
        if mode and hasattr(self, '_frozen_for_finetune') and self._frozen_for_finetune:
            frozen_modules: list[nn.Module] = [
                self.robot_projection,
                self.action_decoder,
                self.proprioceptive_normalizer,
            ]
            for module in frozen_modules:
                module.eval()
    
    def freeze_for_finetune(self):
        """Freeze all modules except human projection for finetuning on SMPLX modality.
        
        This method:
        1. Freezes robot projection, action decoder, and proprioceptive normalizer
        2. Enables gradient computation only for human projection
        3. Goal: optimize human projection to better align with shared latent space
        
        Call this before finetuning with smplx signals.
        """
        self._frozen_for_finetune = True
        
        # Freeze all modules except human_projection
        frozen_modules: list[nn.Module] = [
            self.robot_projection,
            self.action_decoder,
            self.proprioceptive_normalizer,
        ]
        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        # Only human_projection is trainable for finetune
        self.human_projection.train()
        for param in self.human_projection.parameters():
            param.requires_grad = True