import torch
import torch.nn as nn
from typing import Literal
from torch.distributions import Normal
import re

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization


class Actor_Triple_AE(nn.Module):
    """
    Triple Autoencoder for multi-modal robot control with keypoints.
    
    Main network components:
    - Robot encoder-decoder pair: encodes/decodes robot goal state (x_sg)
    - Human encoder-decoder pair: encodes/decodes human state (x_sh)
    - Keypoints encoder-decoder pair: encodes/decodes keypoints SE3 (x_sk)
    - Action decoder: generates actions from latent vector + proprioceptive state
    
    Key differences from Dual_AE:
    1. Three independent encoders for three modalities (robot, human, keypoints)
    2. Three independent decoders for reconstruction
    3. Three-way alignment loss in shared latent space
    4. Cross-modal consistency loss across all three modalities
    
    Main signal flow (when activate_signals = "robot"):
    
    x_sg (robot state) --robot_encoder--> z_robot --robot_decoder--> x_sg_recon
                                              |
                                              ├--[concat w/ x_sp]--action_decoder--> actions
                                              |
    x_sh (human state) --human_encoder--> z_human --human_decoder--> x_sh_recon
                                              |
    x_sk (keypoints) --keypoints_encoder--> z_keypoints --keypoints_decoder--> x_sk_recon
                                              |
    x_sp (proprioceptive) --> EmpiricalNormalization --> concat with z
    
    Loss components (computed externally for integration with RL algorithms):
    - alignment: Three-way MSE alignment (z_sg-z_sh, z_sg-z_sk, z_sh-z_sk)
    - reconstruction: Three MSE terms (x_sg, x_sh, x_sk)
    - consistency: Cross-modal consistency loss based on activate_signals
    """
    
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int,           # robot state dimension
        actor_sh_dim: int = 0,       # human state dimension
        actor_sk_dim: int = 0,       # keypoints state dimension
        # AE Configuration
        latent_dim: int = 32,
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        keypoints_encoder_hidden_dims: list[int] = None,
        robot_decoder_hidden_dims: list[int] = None,
        human_decoder_hidden_dims: list[int] = None,
        keypoints_decoder_hidden_dims: list[int] = None,
        action_decoder_hidden_dims: list[int] = None,
        activation: str = "relu",
        # Modality routing
        activate_signals: Literal["robot", "smplx", "keypoints"] = "robot",
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.actor_sk_dim = actor_sk_dim
        self.num_actions = num_actions
        self.activate_signals = activate_signals
        self.latent_dim = latent_dim
        
        # Set default hidden dimensions if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if keypoints_encoder_hidden_dims is None:
            keypoints_encoder_hidden_dims = [512, 256]
        if robot_decoder_hidden_dims is None:
            robot_decoder_hidden_dims = [256, 512]
        if human_decoder_hidden_dims is None:
            human_decoder_hidden_dims = [256, 512]
        if keypoints_decoder_hidden_dims is None:
            keypoints_decoder_hidden_dims = [256, 512]
        
        activation_fn = resolve_nn_activation(activation)
        
        # ==================== Robot Encoder ====================
        # Input: x_sg (actor_sg_dim) -> Output: latent_dim
        robot_encoder_layers = []
        robot_encoder_layers.append(nn.Linear(actor_sg_dim, robot_encoder_hidden_dims[0]))
        robot_encoder_layers.append(activation_fn)
        for i in range(len(robot_encoder_hidden_dims) - 1):
            robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[i], robot_encoder_hidden_dims[i+1]))
            robot_encoder_layers.append(activation_fn)
        robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[-1], latent_dim))
        self.robot_encoder = nn.Sequential(*robot_encoder_layers)
        
        # ==================== Human Encoder ====================
        # Input: x_sh (actor_sh_dim) -> Output: latent_dim
        human_encoder_layers = []
        human_encoder_layers.append(nn.Linear(actor_sh_dim, human_encoder_hidden_dims[0]))
        human_encoder_layers.append(activation_fn)
        for i in range(len(human_encoder_hidden_dims) - 1):
            human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[i], human_encoder_hidden_dims[i+1]))
            human_encoder_layers.append(activation_fn)
        human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[-1], latent_dim))
        self.human_encoder = nn.Sequential(*human_encoder_layers)
        
        # ==================== Keypoints Encoder ====================
        # Input: x_sk (actor_sk_dim) -> Output: latent_dim
        keypoints_encoder_layers = []
        keypoints_encoder_layers.append(nn.Linear(actor_sk_dim, keypoints_encoder_hidden_dims[0]))
        keypoints_encoder_layers.append(activation_fn)
        for i in range(len(keypoints_encoder_hidden_dims) - 1):
            keypoints_encoder_layers.append(nn.Linear(keypoints_encoder_hidden_dims[i], keypoints_encoder_hidden_dims[i+1]))
            keypoints_encoder_layers.append(activation_fn)
        keypoints_encoder_layers.append(nn.Linear(keypoints_encoder_hidden_dims[-1], latent_dim))
        self.keypoints_encoder = nn.Sequential(*keypoints_encoder_layers)
        
        # ==================== Robot Decoder ====================
        # Input: latent_dim -> Output: x_sg (actor_sg_dim)
        robot_decoder_layers = []
        robot_decoder_layers.append(nn.Linear(latent_dim, robot_decoder_hidden_dims[0]))
        robot_decoder_layers.append(activation_fn)
        for i in range(len(robot_decoder_hidden_dims) - 1):
            robot_decoder_layers.append(nn.Linear(robot_decoder_hidden_dims[i], robot_decoder_hidden_dims[i+1]))
            robot_decoder_layers.append(activation_fn)
        robot_decoder_layers.append(nn.Linear(robot_decoder_hidden_dims[-1], actor_sg_dim))
        self.robot_decoder = nn.Sequential(*robot_decoder_layers)
        
        # ==================== Human Decoder ====================
        # Input: latent_dim -> Output: x_sh (actor_sh_dim)
        human_decoder_layers = []
        human_decoder_layers.append(nn.Linear(latent_dim, human_decoder_hidden_dims[0]))
        human_decoder_layers.append(activation_fn)
        for i in range(len(human_decoder_hidden_dims) - 1):
            human_decoder_layers.append(nn.Linear(human_decoder_hidden_dims[i], human_decoder_hidden_dims[i+1]))
            human_decoder_layers.append(activation_fn)
        human_decoder_layers.append(nn.Linear(human_decoder_hidden_dims[-1], actor_sh_dim))
        self.human_decoder = nn.Sequential(*human_decoder_layers)
        
        # ==================== Keypoints Decoder ====================
        # Input: latent_dim -> Output: x_sk (actor_sk_dim)
        keypoints_decoder_layers = []
        keypoints_decoder_layers.append(nn.Linear(latent_dim, keypoints_decoder_hidden_dims[0]))
        keypoints_decoder_layers.append(activation_fn)
        for i in range(len(keypoints_decoder_hidden_dims) - 1):
            keypoints_decoder_layers.append(nn.Linear(keypoints_decoder_hidden_dims[i], keypoints_decoder_hidden_dims[i+1]))
            keypoints_decoder_layers.append(activation_fn)
        keypoints_decoder_layers.append(nn.Linear(keypoints_decoder_hidden_dims[-1], actor_sk_dim))
        self.keypoints_decoder = nn.Sequential(*keypoints_decoder_layers)
        
        # ==================== Proprioceptive State Normalizer ====================
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim - actor_sk_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # ==================== Action Decoder ====================
        # Input: latent_dim + actor_sp_dim -> Output: num_actions
        action_layers = []
        action_layers.append(nn.Linear(latent_dim + actor_sp_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)
    
    def forward(self, x):
        """Forward pass through the Triple_AE actor with modality routing.
        
        Uses deterministic encoding (no sampling) based on activate_signals.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               Structure: [x_sg | x_sh | x_sk | x_sp]
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        if self.activate_signals == "robot":
            return self.forward_robot(x)
        elif self.activate_signals == "smplx":
            return self.forward_smplx(x)
        elif self.activate_signals == "keypoints":
            return self.forward_keypoints(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot', 'smplx', or 'keypoints'.")
    
    def forward_robot(self, x):
        """Forward pass using robot goal state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               Structure: [x_sg | x_sh | x_sk | x_sp]
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [x_sg | x_sh | x_sk | x_sp]
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # Encode robot state to latent (deterministic, no sampling)
        z_sg = self.robot_encoder(x_sg)
        
        # Detach latent to prevent PPO gradients from flowing back to the encoder
        # Auxiliary losses (reconstruction, alignment, consistency) will train the encoder separately
        z_for_action = z_sg.detach()
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        """Forward pass using SMPLX human state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               Structure: [x_sg | x_sh | x_sk | x_sp]
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [x_sg | x_sh | x_sk | x_sp]
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # Encode human state to latent (deterministic, no sampling)
        z_sh = self.human_encoder(x_sh)
        
        # Detach latent to prevent PPO gradients from flowing back to the encoder
        z_for_action = z_sh.detach()
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_keypoints(self, x):
        """Forward pass using keypoints SE3 modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               Structure: [x_sg | x_sh | x_sk | x_sp]
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [x_sg | x_sh | x_sk | x_sp]
        x_sk = x[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # Encode keypoints state to latent (deterministic, no sampling)
        z_sk = self.keypoints_encoder(x_sk)
        
        # Detach latent to prevent PPO gradients from flowing back to the encoder
        z_for_action = z_sk.detach()
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def encode_robot(self, x):
        """Encode using robot encoder (deterministic).
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sg_dim)
        
        Returns:
            z: Latent vector (deterministic, no sampling)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sg = x[:, :self.actor_sg_dim]
        else:
            x_sg = x
        
        z = self.robot_encoder(x_sg)
        return z
    
    def encode_smplx(self, x):
        """Encode using human encoder (deterministic).
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sh_dim)
        
        Returns:
            z: Latent vector (deterministic, no sampling)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        else:
            x_sh = x
        
        z = self.human_encoder(x_sh)
        return z
    
    def encode_keypoints(self, x):
        """Encode using keypoints encoder (deterministic).
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_sk_dim)
        
        Returns:
            z: Latent vector (deterministic, no sampling)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sk = x[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        else:
            x_sk = x
        
        z = self.keypoints_encoder(x_sk)
        return z
    
    def decode_robot(self, z):
        """Decode latent vector to robot state reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            x_sg_recon: Reconstructed robot state of shape (batch_size, actor_sg_dim)
        """
        return self.robot_decoder(z)
    
    def decode_human(self, z):
        """Decode latent vector to human state reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            x_sh_recon: Reconstructed human state of shape (batch_size, actor_sh_dim)
        """
        return self.human_decoder(z)
    
    def decode_keypoints(self, z):
        """Decode latent vector to keypoints state reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            x_sk_recon: Reconstructed keypoints state of shape (batch_size, actor_sk_dim)
        """
        return self.keypoints_decoder(z)
    
    def decode_action(self, z, x_sp):
        """Decode actions from latent and proprioceptive state.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.action_decoder(torch.cat([z, x_sp_normalized], dim=-1))
    
    def forward_robot_exporter(self, robot_command, proprioceptive_state):
        """Forward pass for deployment with decoupled inputs.
        
        Args:
            robot_command: Input tensor of shape (batch_size, actor_sg_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        z_sg = self.robot_encoder(robot_command)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sg, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def forward_smplx_exporter(self, smplx_human_state, proprioceptive_state):
        """Forward pass for deployment with decoupled inputs.
        
        Args:
            smplx_human_state: Input tensor of shape (batch_size, actor_sh_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        z_sh = self.human_encoder(smplx_human_state)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sh, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def forward_keypoints_exporter(self, keypoints_state, proprioceptive_state):
        """Forward pass for deployment with decoupled inputs.
        
        Args:
            keypoints_state: Input tensor of shape (batch_size, actor_sk_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        z_sk = self.keypoints_encoder(keypoints_state)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sk, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def freeze_cmd_encoder(self):
        """Freeze all encoders and decoders, keep action decoder trainable.
        
        This method freezes all three encoder-decoder pairs (robot, human, keypoints)
        while keeping the action decoder trainable. Used when finetuning for downstream tasks.
        """
        # List of all encoders and decoders
        encoders_decoders = [
            self.robot_encoder, self.human_encoder, self.keypoints_encoder,
            self.robot_decoder, self.human_decoder, self.keypoints_decoder,
        ]
        
        # Freeze all encoders and decoders
        for module in encoders_decoders:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        # Keep action decoder trainable
        for param in self.action_decoder.parameters():
            param.requires_grad = True
        self.action_decoder.train()
        
        print(f"[INFO] Actor_Triple_AE: Frozen all encoders and decoders")
        print(f"       - Frozen: robot_encoder, human_encoder, keypoints_encoder")
        print(f"       -         robot_decoder, human_decoder, keypoints_decoder")
        print(f"       - Trainable: action_decoder, proprioceptive_normalizer")
    
    
    def freeze_for_finetune(self, finetune_networks: list[str] = ['human_encoder', 'human_decoder']):
        """Freeze all networks except those specified for finetuning.
        
        Args:
            finetune_networks: List of network names to keep trainable.
                               Options: 'robot_encoder', 'human_encoder', 'keypoints_encoder',
                                        'robot_decoder', 'human_decoder', 'keypoints_decoder',
                                        'action_decoder', 'proprioceptive_normalizer'
        """
        self._frozen_for_finetune = True
        
        all_networks = {
            'robot_encoder': self.robot_encoder,
            'human_encoder': self.human_encoder,
            'keypoints_encoder': self.keypoints_encoder,
            'robot_decoder': self.robot_decoder,
            'human_decoder': self.human_decoder,
            'keypoints_decoder': self.keypoints_decoder,
            'action_decoder': self.action_decoder,
            'proprioceptive_normalizer': self.proprioceptive_normalizer,
        }
        
        invalid_networks = set(finetune_networks) - set(all_networks.keys())
        if invalid_networks:
            raise ValueError(f"Invalid network names: {invalid_networks}")
        
        for network_name, network_module in all_networks.items():
            for param in network_module.parameters():
                param.requires_grad = False
            network_module.eval()
        
        for network_name in finetune_networks:
            for param in all_networks[network_name].parameters():
                param.requires_grad = True
            all_networks[network_name].train()
        
        print(f"[INFO] Actor_Triple_AE: Frozen networks for finetuning")
        print(f"       - Trainable: {finetune_networks}")
        print(f"       - Frozen: {[n for n in all_networks.keys() if n not in finetune_networks]}")
    
    def train(self, mode: bool = True):
        """Override train() to manage frozen modules during finetuning."""
        super().train(mode)


class Critic_Triple_AE(nn.Module):
    """Critic network for Triple_AE-based actor-critic architecture.
    
    This is an MLP critic that takes a latent vector from Triple_AE encoder
    and proprioceptive state to estimate value.
    
    Network Architecture:
        Input: latent_vector (latent_dim) + proprioceptive_state (critic_sp_dim)
        Hidden layers: critic_hidden_dims
        Output: value estimate (scalar)
    """
    
    def __init__(
        self,
        latent_dim: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = None,
        activation: str = "elu",
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.critic_sp_dim = critic_sp_dim
        
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256]
        
        activation_fn = resolve_nn_activation(activation)
        
        # Proprioceptive State Normalizer
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(critic_sp_dim,))
        
        # Critic network: [latent + proprioceptive_state] -> value
        critic_input_dim = latent_dim + critic_sp_dim
        critic_layers = []
        
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic_mlp = nn.Sequential(*critic_layers)
    
    def forward(self, z, x_sp):
        """Forward pass of the critic.
        
        Args:
            z: Latent vector from Triple_AE encoder of shape (batch_size, latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, critic_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.critic_mlp(torch.cat([z, x_sp_normalized], dim=-1))


class ActorCritic_Triple_AE(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # actor triple_ae specific
        actor_sg_dim: int = None,
        actor_sh_dim: int = 0,
        actor_sk_dim: int = 0,
        latent_dim: int = 32,
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        keypoints_encoder_hidden_dims: list[int] = None,
        robot_decoder_hidden_dims: list[int] = None,
        human_decoder_hidden_dims: list[int] = None,
        keypoints_decoder_hidden_dims: list[int] = None,
        activate_signals: Literal["robot", "smplx", "keypoints"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(f"[WARNING] Unexpected kwargs: {kwargs}")
        
        super().__init__()
        activation_fn = resolve_nn_activation(activation)
        
        # Set default hidden dims if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if keypoints_encoder_hidden_dims is None:
            keypoints_encoder_hidden_dims = [512, 256]
        if robot_decoder_hidden_dims is None:
            robot_decoder_hidden_dims = [256, 512]
        if human_decoder_hidden_dims is None:
            human_decoder_hidden_dims = [256, 512]
        if keypoints_decoder_hidden_dims is None:
            keypoints_decoder_hidden_dims = [256, 512]
        
        self.activate_signals = activate_signals
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.actor_sk_dim = actor_sk_dim
        
        # ==================== Actor: Triple_AE Policy ====================
        self.actor = Actor_Triple_AE(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            actor_sh_dim=actor_sh_dim,
            actor_sk_dim=actor_sk_dim,
            latent_dim=latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            keypoints_encoder_hidden_dims=keypoints_encoder_hidden_dims,
            robot_decoder_hidden_dims=robot_decoder_hidden_dims,
            human_decoder_hidden_dims=human_decoder_hidden_dims,
            keypoints_decoder_hidden_dims=keypoints_decoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            activation=activation,
            activate_signals=activate_signals,
        )
        
        # ==================== Critic: Value function ====================
        critic_sp_dim = num_critic_obs - actor_sg_dim - actor_sh_dim - actor_sk_dim
        
        self.critic = Critic_Triple_AE(
            latent_dim=latent_dim,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # ==================== Action noise ====================
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
        
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update the distribution based on observations."""
        mean = self.actor(observations)
        
        if self.noise_std_type == "scalar":
            std = torch.exp(self.log_std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value from critic using activate_signals to choose modality."""
        x_sp_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        if self.activate_signals == "robot":
            x_sg_critic = critic_observations[:, :self.actor_sg_dim]
            z = self.actor.encode_robot(x_sg_critic)
        elif self.activate_signals == "smplx":
            x_sh_critic = critic_observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
            z = self.actor.encode_smplx(x_sh_critic)
        elif self.activate_signals == "keypoints":
            x_sk_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
            z = self.actor.encode_keypoints(x_sk_critic)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}")
        
        # Detach latent to prevent critic gradients from flowing back to the encoder
        # The encoder is updated through auxiliary losses (reconstruction, alignment, consistency)
        z = z.detach()
        
        value = self.critic(z, x_sp_critic)
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def get_alignment_loss(self, observations):
        """Compute three-way latent space alignment loss.
        
        Alignment loss = w_sg_sh * MSE(z_sg, z_sh) + w_sg_sk * MSE(z_sg, z_sk) + w_sh_sk * MSE(z_sh, z_sk)
        
        Args:
            observations: Input observations containing robot, human, keypoints states and proprioception.
        
        Returns:
            Dictionary containing individual alignment losses:
            {
                'alignment_sg_sh': MSE between robot and human latent vectors,
                'alignment_sg_sk': MSE between robot and keypoints latent vectors,
                'alignment_sh_sk': MSE between human and keypoints latent vectors,
                'alignment_total': Weighted sum of all alignment losses
            }
        """
        # Encode all three modalities
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        # Compute three-way alignment losses
        align_sg_sh = torch.mean((z_sg - z_sh) ** 2)
        align_sg_sk = torch.mean((z_sg - z_sk) ** 2)
        align_sh_sk = torch.mean((z_sh - z_sk) ** 2)
        
        # Weighted combination (you can adjust weights)
        alignment_total = align_sg_sh + 0.8 * align_sg_sk + 0.6 * align_sh_sk
        
        return {
            'alignment_sg_sh': align_sg_sh,
            'alignment_sg_sk': align_sg_sk,
            'alignment_sh_sk': align_sh_sk,
            'alignment_total': alignment_total
        }
    
    def get_reconstruction_loss(self, observations):
        """Compute reconstruction losses for all three decoders.
        
        Args:
            observations: Input observations of shape (batch_size, num_critic_obs)
        
        Returns:
            Dictionary containing:
            {
                'recon_sg': MSE reconstruction for robot state,
                'recon_sh': MSE reconstruction for human state,
                'recon_sk': MSE reconstruction for keypoints state,
                'recon_total': Sum of all reconstruction losses
            }
        """
        # Extract state components
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        # Encode all three modalities
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        # Reconstruct and compute losses
        x_sg_recon = self.actor.decode_robot(z_sg)
        recon_sg = torch.mean((x_sg_recon - x_sg) ** 2)
        
        x_sh_recon = self.actor.decode_human(z_sh)
        recon_sh = torch.mean((x_sh_recon - x_sh) ** 2)
        
        x_sk_recon = self.actor.decode_keypoints(z_sk)
        recon_sk = torch.mean((x_sk_recon - x_sk) ** 2)
        
        recon_total = recon_sg + recon_sh + recon_sk
        
        return {
            'recon_sg': recon_sg,
            'recon_sh': recon_sh,
            'recon_sk': recon_sk,
            'recon_total': recon_total
        }
    
    def get_consistency_loss(self, observations):
        """Compute cross-modal consistency loss for Triple_AE.
        
        This loss ensures consistency by encoding in one modality and decoding with 
        another modality's decoder, verifying that the latent space is semantically aligned.
        
        When activate_signals == "robot":
            - Encode human and keypoints states
            - Decode with robot_decoder
            - Loss = MSE(decoder_robot(z_sh), x_sg) + MSE(decoder_robot(z_sk), x_sg)
            
        When activate_signals == "smplx":
            - Encode robot and keypoints states
            - Decode with human_decoder
            - Loss = MSE(decoder_human(z_sg), x_sh) + MSE(decoder_human(z_sk), x_sh)
            
        When activate_signals == "keypoints":
            - Encode robot and human states
            - Decode with keypoints_decoder
            - Loss = MSE(decoder_keypoints(z_sg), x_sk) + MSE(decoder_keypoints(z_sh), x_sk)
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
        
        Returns:
            Dictionary containing:
            {
                'consistency_loss1': First cross-modal consistency loss,
                'consistency_loss2': Second cross-modal consistency loss,
                'consistency_total': Sum of both losses
            }
        """
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        mse_loss = torch.nn.MSELoss()
        
        if self.activate_signals == "robot":
            # Decode human and keypoints latents with robot decoder
            x_sg_from_human = self.actor.decode_robot(z_sh)
            x_sg_from_keypoints = self.actor.decode_robot(z_sk)
            
            consist_1 = mse_loss(x_sg_from_human, x_sg)
            consist_2 = mse_loss(x_sg_from_keypoints, x_sg)
            consist_total = consist_1 + consist_2
            
        elif self.activate_signals == "smplx":
            # Decode robot and keypoints latents with human decoder
            x_sh_from_robot = self.actor.decode_human(z_sg)
            x_sh_from_keypoints = self.actor.decode_human(z_sk)
            
            consist_1 = mse_loss(x_sh_from_robot, x_sh)
            consist_2 = mse_loss(x_sh_from_keypoints, x_sh)
            consist_total = consist_1 + consist_2
            
        elif self.activate_signals == "keypoints":
            # Decode robot and human latents with keypoints decoder
            x_sk_from_robot = self.actor.decode_keypoints(z_sg)
            x_sk_from_human = self.actor.decode_keypoints(z_sh)
            
            consist_1 = mse_loss(x_sk_from_robot, x_sk)
            consist_2 = mse_loss(x_sk_from_human, x_sk)
            consist_total = consist_1 + consist_2
            
        else:
            raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
        
        return {
            'consistency_loss1': consist_1,
            'consistency_loss2': consist_2,
            'consistency_total': consist_total
        }
    
    def get_auxiliary_loss(self, observations, loss_items: list[str] = ['alignment', 'reconstruction', 'consistency']):
        """Compute all specified losses for Triple_AE actor-critic in one pass.
        
        This method efficiently computes all requested losses by performing
        encoding only once and computing all losses from shared representations.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
                         containing [x_sg | x_sh | x_sk | x_sp]
            loss_items: List of loss types to compute.
                       Options: 'alignment', 'reconstruction', 'consistency'
                       Default: ['alignment', 'reconstruction', 'consistency']
        
        Returns:
            Dictionary with computed losses with keys:
            {
                'alignment': Three-way MSE alignment loss (weighted sum),
                'reconstruction_sg': Reconstruction loss for robot state,
                'reconstruction_sh': Reconstruction loss for human state,
                'reconstruction_sk': Reconstruction loss for keypoints state,
                'consistency': Cross-modal consistency loss,
            }
        """
        losses = {
            "alignment": None,
            "reconstruction_sg": None,
            "reconstruction_sh": None,
            "reconstruction_sk": None,
            "consistency": None
        }
        
        # Extract states once
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        # Encode all modalities once
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        mse_loss = torch.nn.MSELoss()
        
        # Compute reconstruction losses if requested
        if 'reconstruction' in loss_items:
            x_sg_recon = self.actor.decode_robot(z_sg)
            recon_sg = torch.mean((x_sg_recon - x_sg) ** 2)
            losses['reconstruction_sg'] = recon_sg
            
            x_sh_recon = self.actor.decode_human(z_sh)
            recon_sh = torch.mean((x_sh_recon - x_sh) ** 2)
            losses['reconstruction_sh'] = recon_sh
            
            x_sk_recon = self.actor.decode_keypoints(z_sk)
            recon_sk = torch.mean((x_sk_recon - x_sk) ** 2)
            losses['reconstruction_sk'] = recon_sk
        
        # Compute alignment loss if requested
        if 'alignment' in loss_items:
            align_sg_sh = torch.mean((z_sg - z_sh) ** 2)
            align_sg_sk = torch.mean((z_sg - z_sk) ** 2)
            align_sh_sk = torch.mean((z_sh - z_sk) ** 2)
            
            # Weighted three-way alignment (using actor's alignment weights)
            alignment_loss = align_sg_sh + align_sg_sk + align_sh_sk
            losses['alignment'] = alignment_loss
        
        # Compute consistency loss if requested
        if 'consistency' in loss_items:
            if self.activate_signals == "robot":
                x_sg_from_human = self.actor.decode_robot(z_sh)
                x_sg_from_keypoints = self.actor.decode_robot(z_sk)
                consistency_loss = (
                    mse_loss(x_sg_from_human, x_sg) +
                    mse_loss(x_sg_from_keypoints, x_sg)
                ) / 2.0
            elif self.activate_signals == "smplx":
                x_sh_from_robot = self.actor.decode_human(z_sg)
                x_sh_from_keypoints = self.actor.decode_human(z_sk)
                consistency_loss = (
                    mse_loss(x_sh_from_robot, x_sh) +
                    mse_loss(x_sh_from_keypoints, x_sh)
                ) / 2.0
            elif self.activate_signals == "keypoints":
                x_sk_from_robot = self.actor.decode_keypoints(z_sg)
                x_sk_from_human = self.actor.decode_keypoints(z_sh)
                consistency_loss = (
                    mse_loss(x_sk_from_robot, x_sk) +
                    mse_loss(x_sk_from_human, x_sk)
                ) / 2.0
            else:
                raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
            
            losses['consistency'] = consistency_loss
        
        return losses

###########################################
# ActorCritic_Triple_AE_Single_Finetune 
#   A variant of ActorCritic_Triple_AE for single modality finetuning.
#   usage: `freeze` parameter to freeze cmd encoders/decoders only. finetune the action decoder. 
###########################################

class Actor_Triple_AE_Single_Finetune(nn.Module):
    """
    Single modality Actor for finetuning with frozen encoder/decoder.
    
    This actor uses a single cmd encoder (pre-trained) and action decoder.
    During finetuning, the cmd encoder can be frozen while only training the action decoder.
    
    Network Architecture:
        cmd_state (actor_cmd_dim) -> cmd_encoder -> latent_dim
        latent_dim + proprioceptive_state -> action_decoder -> actions
        latent_dim -> cmd_decoder -> cmd_state_recon (for optional reconstruction loss)
    """
    
    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_cmd_dim: int,
        latent_dim: int = 32,
        cmd_encoder_hidden_dims: list[int] = None,
        cmd_decoder_hidden_dims: list[int] = None,
        action_decoder_hidden_dims: list[int] = None,
        activation: str = "elu",
        freeze: bool = True,
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_cmd_dim = actor_cmd_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.freeze = freeze
        
        # Set default hidden dimensions if not provided
        if cmd_encoder_hidden_dims is None:
            cmd_encoder_hidden_dims = [512, 256]
        if cmd_decoder_hidden_dims is None:
            cmd_decoder_hidden_dims = [256, 512]
        if action_decoder_hidden_dims is None:
            action_decoder_hidden_dims = [256, 256, 256]
        
        activation_fn = resolve_nn_activation(activation)
        
        # ==================== Cmd Encoder ====================
        # Input: cmd_state (actor_cmd_dim) -> Output: latent_dim
        cmd_encoder_layers = []
        cmd_encoder_layers.append(nn.Linear(actor_cmd_dim, cmd_encoder_hidden_dims[0]))
        cmd_encoder_layers.append(activation_fn)
        for i in range(len(cmd_encoder_hidden_dims) - 1):
            cmd_encoder_layers.append(nn.Linear(cmd_encoder_hidden_dims[i], cmd_encoder_hidden_dims[i+1]))
            cmd_encoder_layers.append(activation_fn)
        cmd_encoder_layers.append(nn.Linear(cmd_encoder_hidden_dims[-1], latent_dim))
        self.cmd_encoder = nn.Sequential(*cmd_encoder_layers)
        
        # ==================== Cmd Decoder ====================
        # Input: latent_dim -> Output: cmd_state (actor_cmd_dim)
        cmd_decoder_layers = []
        cmd_decoder_layers.append(nn.Linear(latent_dim, cmd_decoder_hidden_dims[0]))
        cmd_decoder_layers.append(activation_fn)
        for i in range(len(cmd_decoder_hidden_dims) - 1):
            cmd_decoder_layers.append(nn.Linear(cmd_decoder_hidden_dims[i], cmd_decoder_hidden_dims[i+1]))
            cmd_decoder_layers.append(activation_fn)
        cmd_decoder_layers.append(nn.Linear(cmd_decoder_hidden_dims[-1], actor_cmd_dim))
        self.cmd_decoder = nn.Sequential(*cmd_decoder_layers)
        
        # ==================== Proprioceptive State Normalizer ====================
        actor_sp_dim = num_actor_obs - actor_cmd_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # ==================== Action Decoder ====================
        # Input: latent_dim + actor_sp_dim -> Output: num_actions
        action_layers = []
        action_layers.append(nn.Linear(latent_dim + actor_sp_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)
        
        # Apply freeze if specified
        if freeze:
            self._apply_freeze()
    
    def _apply_freeze(self):
        """Freeze cmd encoder and decoder, keep action decoder trainable."""
        # Freeze cmd encoder
        for param in self.cmd_encoder.parameters():
            param.requires_grad = False
        self.cmd_encoder.eval()
        
        # Freeze cmd decoder
        for param in self.cmd_decoder.parameters():
            param.requires_grad = False
        self.cmd_decoder.eval()
        
        print(f"[INFO] Actor_Triple_AE_Single_Finetune: Frozen cmd_encoder and cmd_decoder")
        print(f"       - Trainable: action_decoder, proprioceptive_normalizer")
    
    def forward(self, x):
        """Forward pass through the single modality actor.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
               Structure: [cmd_state | proprioceptive_state]
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [cmd_state | proprioceptive_state]
        x_cmd = x[:, :self.actor_cmd_dim]
        x_sp = x[:, self.actor_cmd_dim:]
        
        # Encode cmd state to latent
        z = self.cmd_encoder(x_cmd)
        # Detach latent so PPO gradients do not update the encoder during actor loss
        z_for_action = z.detach()
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def encode(self, x):
        """Encode cmd state to latent.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs) or (batch_size, actor_cmd_dim)
        
        Returns:
            z: Latent vector of shape (batch_size, latent_dim)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_cmd = x[:, :self.actor_cmd_dim]
        else:
            x_cmd = x
        
        return self.cmd_encoder(x_cmd)
    
    def decode_cmd(self, z):
        """Decode latent to cmd state reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            cmd_recon: Reconstructed cmd state of shape (batch_size, actor_cmd_dim)
        """
        return self.cmd_decoder(z)
    
    def decode_action(self, z, x_sp):
        """Decode actions from latent and proprioceptive state.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        # Use detached latent to keep action decoding gradients out of the encoder
        return self.action_decoder(torch.cat([z.detach(), x_sp_normalized], dim=-1))
    
    def forward_exporter(self, cmd_state, proprioceptive_state):
        """Forward pass for deployment with decoupled inputs.
        
        Args:
            cmd_state: Input tensor of shape (batch_size, actor_cmd_dim)
            proprioceptive_state: Input tensor of shape (batch_size, actor_sp_dim)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        z = self.cmd_encoder(cmd_state)
        z_for_action = z.detach()
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_for_action, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def train(self, mode: bool = True):
        """Override train() to keep frozen modules in eval mode."""
        super().train(mode)
        if self.freeze:
            self.cmd_encoder.eval()
            self.cmd_decoder.eval()
        return self


class ActorCritic_Triple_AE_Single_Finetune(nn.Module):
    """
    ActorCritic for single modality finetuning.
    
    This variant freezes the cmd encoder/decoder and only finetunes the action decoder.
    Useful for adapting a pre-trained multi-modal model to a single modality task.
    
    The critic uses the same latent space from the frozen encoder.
    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # actor triple_ae specific
        actor_cmd_dim: int = None,
        latent_dim: int = 32,
        cmd_encoder_hidden_dims: list[int] = None,
        cmd_decoder_hidden_dims: list[int] = None,
        freeze: bool = True,
        encoder_key: Literal["robot", "human", "keypoints"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(f"[WARNING] Unexpected kwargs: {kwargs}")
        
        super().__init__()
        
        self.actor_cmd_dim = actor_cmd_dim
        self.latent_dim = latent_dim
        self.freeze = freeze
        self.encoder_key = encoder_key
        
        # Set default hidden dims if not provided
        if cmd_encoder_hidden_dims is None:
            cmd_encoder_hidden_dims = [512, 256]
        if cmd_decoder_hidden_dims is None:
            cmd_decoder_hidden_dims = [256, 512]
        
        # ==================== Actor: Single Modality AE Policy ====================
        self.actor = Actor_Triple_AE_Single_Finetune(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_cmd_dim=actor_cmd_dim,
            latent_dim=latent_dim,
            cmd_encoder_hidden_dims=cmd_encoder_hidden_dims,
            cmd_decoder_hidden_dims=cmd_decoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            activation=activation,
            freeze=freeze,
        )
        
        # ==================== Critic: Value function ====================
        critic_sp_dim = num_critic_obs - actor_cmd_dim
        
        self.critic = Critic_Triple_AE(
            latent_dim=latent_dim,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # ==================== Action noise ====================
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update the distribution based on observations."""
        mean = self.actor(observations)
        
        if self.noise_std_type == "scalar":
            std = torch.exp(self.log_std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value from critic."""
        x_cmd = critic_observations[:, :self.actor_cmd_dim]
        x_sp_critic = critic_observations[:, self.actor_cmd_dim:]
        
        z = self.actor.encode(x_cmd)
        # Detach latent to prevent critic gradients from flowing back to the encoder
        # When encoder is frozen, this has no effect; when unfrozen, it separates gradient paths
        z = z.detach()
        value = self.critic(z, x_sp_critic)
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def get_reconstruction_loss(self, observations):
        """Compute reconstruction loss for cmd state.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
        
        Returns:
            Dictionary containing:
            {
                'recon_cmd': MSE reconstruction for cmd state,
            }
        """
        x_cmd = observations[:, :self.actor_cmd_dim]
        
        z = self.actor.encode(observations)
        x_cmd_recon = self.actor.decode_cmd(z)
        
        recon_cmd = torch.mean((x_cmd_recon - x_cmd) ** 2)
        
        return {
            'recon_cmd': recon_cmd,
        }
    
    def get_auxiliary_loss(self, observations, loss_items: list[str] = ['reconstruction']):
        """Compute auxiliary losses for single modality finetuning.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
            loss_items: List of loss types to compute.
                       Options: 'reconstruction'
        
        Returns:
            Dictionary with computed losses.
        """
        losses = {
            "reconstruction_cmd": None,
        }
        
        if 'reconstruction' in loss_items:
            x_cmd = observations[:, :self.actor_cmd_dim]
            z = self.actor.encode(observations)
            x_cmd_recon = self.actor.decode_cmd(z)
            losses['reconstruction_cmd'] = torch.mean((x_cmd_recon - x_cmd) ** 2)
        
        return losses
    
    def load_state_dict(self, state_dict, strict=True):
        """Load pretrained weights from ActorCritic_Triple_AE model.
        
        Maps the selected encoder/decoder to cmd_encoder/cmd_decoder and
        removes unused encoder/decoder weights.
        
        Args:
            state_dict: State dict from a pre-trained ActorCritic_Triple_AE model.
            strict: Whether to strictly enforce that the keys match.
        """
        encoder_key = f"{self.encoder_key}_encoder"
        decoder_key = f"{self.encoder_key}_decoder"
        
        # All encoder/decoder keys in Triple_AE
        all_encoder_keys = ["robot_encoder", "human_encoder", "keypoints_encoder"]
        all_decoder_keys = ["robot_decoder", "human_decoder", "keypoints_decoder"]
        
        # Keys to delete (encoders/decoders we don't need)
        delete_encoder_keys = [k for k in all_encoder_keys if k != encoder_key]
        delete_decoder_keys = [k for k in all_decoder_keys if k != decoder_key]
        
        # Build new state dict with remapped keys
        # Patterns for keys to delete
        delete_patterns = [re.compile(rf'actor\.{k}\.') for k in delete_encoder_keys + delete_decoder_keys]
        
        # Patterns for keys to remap
        encoder_pattern = re.compile(rf'(actor\.){encoder_key}(\..*)')
        decoder_pattern = re.compile(rf'(actor\.){decoder_key}(\..*)')
        
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip keys from unused encoders/decoders
            if any(p.search(key) for p in delete_patterns):
                continue
            
            # Remap selected encoder to cmd_encoder
            match = encoder_pattern.match(key)
            if match:
                new_key = f"{match.group(1)}cmd_encoder{match.group(2)}"
                new_state_dict[new_key] = value
                continue
            
            # Remap selected decoder to cmd_decoder
            match = decoder_pattern.match(key)
            if match:
                new_key = f"{match.group(1)}cmd_decoder{match.group(2)}"
                new_state_dict[new_key] = value
                continue
            
            # Keep other keys as-is
            new_state_dict[key] = value
        
        # Load the remapped state dict
        super().load_state_dict(new_state_dict, strict=strict)
        
        print(f"[INFO] Loaded pretrained weights with encoder_key='{self.encoder_key}'")
        print(f"       - Mapped: actor.{encoder_key} -> actor.cmd_encoder")
        print(f"       - Mapped: actor.{decoder_key} -> actor.cmd_decoder")
        print(f"       - Deleted: {delete_encoder_keys + delete_decoder_keys}")
        
        # Re-apply freeze after loading
        if self.freeze:
            self.actor._apply_freeze()
        
        return True
        
        