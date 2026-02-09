import torch
import torch.nn as nn
from typing import Literal
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

class Actor_Dual_AE(nn.Module):
    """
    Dual Autoencoder for multi-modal robot control.
    
    Main network components:
    - Robot encoder-decoder pair: encodes/decodes robot goal state (x_sg)
    - Human encoder-decoder pair: encodes/decodes human state (x_sh)
    - Action decoder: generates actions from latent vector + proprioceptive state
    
    Key differences from VAE:
    1. No reparameterization: uses deterministic encoding (latent = encoder output)
    2. Dual encoders with independent latent spaces that are aligned via MSE loss
    3. Dual decoders: robot_decoder reconstructs x_sg, human_decoder reconstructs x_sh
    4. No sampling during inference or training
    
    Main signal flow (when activate_signals = "robot"):
    
    x_sg (robot state) --robot_encoder--> z_robot --robot_decoder--> x_sg_recon
                                              |
                                              ├--[concat w/ x_sp]--action_decoder--> actions
                                              |
    x_sh (human state) --human_encoder--> z_human --human_decoder--> x_sh_recon
                                              |
    x_sp (proprioceptive) --> EmpiricalNormalization --> concat with z
    
    Loss components (computed externally for integration with RL algorithms):
    - recon_robot = MSE(x_sg_recon, x_sg)
    - recon_human = MSE(x_sh_recon, x_sh)
    - align = MSE(z_robot, z_human)  # Latent space alignment
    """
    
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int,  # robot state dimension
        actor_sh_dim: int = 0,  # human state dimension (0 for robot-only mode)
        # AE Configuration
        latent_dim: int = 32,
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = [512, 256],
        human_encoder_hidden_dims: list[int] = [512, 256],
        robot_decoder_hidden_dims: list[int] = [256, 512],
        human_decoder_hidden_dims: list[int] = [256, 512],
        action_decoder_hidden_dims: list[int] = [256, 128],
        activation: str = "relu",
        # Modality routing
        activate_signals: Literal["robot", "smplx"] = "robot",
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.num_actions = num_actions
        self.activate_signals = activate_signals
        self.latent_dim = latent_dim
        
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
        
        # ==================== Proprioceptive State Normalizer ====================
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim
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
        """Forward pass through the Dual_AE actor with modality routing.
        
        Uses deterministic encoding (no sampling) based on activate_signals.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        if self.activate_signals == "robot":
            return self.forward_robot(x)
        elif self.activate_signals == "smplx":
            return self.forward_smplx(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot' or 'smplx'.")
    
    def forward_robot(self, x):
        """Forward pass using robot goal state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [x_sg | x_sh | x_sp]
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Encode robot state to latent (deterministic, no sampling)
        z_sg = self.robot_encoder(x_sg)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sg, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        """Forward pass using SMPLX human state modality.
        
        Args:
            x: Input tensor of shape (batch_size, num_actor_obs)
        
        Returns:
            actions: Predicted actions of shape (batch_size, num_actions)
        """
        # Split input: [x_sg | x_sh | x_sp]
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Encode human state to latent (deterministic, no sampling)
        z_sh = self.human_encoder(x_sh)
        
        # Normalize proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, x_sp_normalized], dim=-1))
        
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
        # Encode robot state to latent
        z_sg = self.robot_encoder(robot_command)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
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
        # Encode human state to latent
        z_sh = self.human_encoder(smplx_human_state)
        
        # Normalize proprioceptive state
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        
        # Decode actions
        actions = self.action_decoder(torch.cat([z_sh, proprioceptive_state_normalized], dim=-1))
        
        return actions
    
    def train(self, mode: bool = True):
        """Override train() to manage frozen modules during finetuning.
        
        The proprioceptive_normalizer is kept in training mode to continuously
        update its normalization statistics.
        
        Args:
            mode (bool): Whether to set the module to training mode.
        """
        super().train(mode)
        # Ensure frozen modules remain in eval mode if mode=True and we're finetuning
        if mode and hasattr(self, '_frozen_for_finetune') and self._frozen_for_finetune:
            frozen_modules: list[nn.Module] = [
                self.robot_encoder,
                self.robot_decoder,
                self.action_decoder,
                self.proprioceptive_normalizer,
            ]
            for module in frozen_modules:
                module.eval()
    
    def freeze_for_finetune(self, finetune_networks: list[str] = ['human_encoder', 'human_decoder']):
        """Freeze all networks except those specified for finetuning.
        
        This method enables selective finetuning by freezing non-trainable networks
        and enabling gradients only for specified networks.
        
        Args:
            finetune_networks: List of network names to keep trainable.
                               Options: 'robot_encoder', 'human_encoder',
                                        'robot_decoder', 'human_decoder',
                                        'action_decoder', 'proprioceptive_normalizer'
        
        Example:
            # Finetune only human encoder and decoder
            actor.freeze_for_finetune(['human_encoder', 'human_decoder'])
            
            # Finetune only human encoder
            actor.freeze_for_finetune(['human_encoder'])
        """
        self._frozen_for_finetune = True
        
        # All available networks
        all_networks = {
            'robot_encoder': self.robot_encoder,
            'human_encoder': self.human_encoder,
            'robot_decoder': self.robot_decoder,
            'human_decoder': self.human_decoder,
            'action_decoder': self.action_decoder,
            'proprioceptive_normalizer': self.proprioceptive_normalizer,
        }
        
        # Validate input network names
        invalid_networks = set(finetune_networks) - set(all_networks.keys())
        if invalid_networks:
            raise ValueError(
                f"Invalid network names: {invalid_networks}. "
                f"Valid options are: {list(all_networks.keys())}"
            )
        
        # Freeze all networks first
        for network_name, network_module in all_networks.items():
            network_module.eval()
            for param in network_module.parameters():
                param.requires_grad = False
        
        # Unfreeze only specified networks for finetuning
        for network_name in finetune_networks:
            network_module = all_networks[network_name]
            network_module.train()
            for param in network_module.parameters():
                param.requires_grad = True
        
        # Print finetuning configuration
        print(f"[INFO] Actor_Dual_AE: Frozen networks for finetuning")
        print(f"       - Trainable: {finetune_networks}")
        print(f"       - Frozen: {[n for n in all_networks.keys() if n not in finetune_networks]}") 
    

 
class Critic_Dual_AE(nn.Module):
    """Critic network for Dual_AE-based actor-critic architecture.
    
    This is an MLP critic that takes a latent vector from Dual_AE encoder
    and proprioceptive state to estimate value.
    
    Network Architecture:
        Input: latent_vector (latent_dim) + proprioceptive_state (critic_sp_dim)
        Hidden layers: critic_hidden_dims
        Output: value estimate (scalar)
        
    Proprioceptive State Normalization:
        - Only the proprioceptive state (x_p) is normalized via EmpiricalNormalization
        - The latent vector from encoder is NOT normalized (it comes from encoder output)
    """
    
    def __init__(
        self,
        latent_dim: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.critic_sp_dim = critic_sp_dim
        
        activation_fn = resolve_nn_activation(activation)
        
        # Proprioceptive State Normalizer (only for x_p, not for latent space)
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
            z: Latent vector from Dual_AE encoder of shape (batch_size, latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, critic_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        # Normalize only the proprioceptive state
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # Concatenate latent and normalized proprioceptive state
        return self.critic_mlp(torch.cat([z, x_sp_normalized], dim=-1))
    
    
class ActorCritic_Dual_AE(nn.Module):
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
        # actor dual_ae specific
        actor_sg_dim: int = None,
        actor_sh_dim: int = 0,  # smplx human state dimension
        latent_dim: int = 32,
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        robot_decoder_hidden_dims: list[int] = None,
        human_decoder_hidden_dims: list[int] = None,
        activate_signals: Literal["robot", "smplx"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic_Dual_AE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = resolve_nn_activation(activation)
        
        # Set default hidden dims if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if robot_decoder_hidden_dims is None:
            robot_decoder_hidden_dims = [256, 512]
        if human_decoder_hidden_dims is None:
            human_decoder_hidden_dims = [256, 512]
        
        self.activate_signals = activate_signals
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim

        #####################################################################################
        # Actor: Dual_AE Policy (deterministic encoder, no sampling)
        # Signal flow:
        #    x^g --robot_encoder--> z^g --robot_decoder--> x^g_recon
        #                             |
        #                             ├──[concat]── action_decoder── action
        #    x^p ──────────────┘
        #
        #    x^h --human_encoder--> z^h --human_decoder--> x^h_recon
        #                             |
        #                             ├── (for alignment loss: MSE(z^g, z^h))
        #####################################################################################
        
        self.actor = Actor_Dual_AE(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            actor_sh_dim=actor_sh_dim,
            latent_dim=latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            robot_decoder_hidden_dims=robot_decoder_hidden_dims,
            human_decoder_hidden_dims=human_decoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            activation=activation,
            activate_signals=activate_signals,
        )

        #####################################################################################
        # Critic: Value function
        # Signal flow:
        #     z (from actor encoder) ──┐
        #                              ├──critic_mlp── V(s)
        #     x^p (proprioceptive) ────┘
        #####################################################################################
        
        # Calculate critic proprioceptive dimension
        critic_sp_dim = num_critic_obs - actor_sg_dim - actor_sh_dim
        
        self.critic = Critic_Dual_AE(
            latent_dim=latent_dim,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
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
        # compute mean
        mean = self.actor(observations)
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
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
        """Evaluate value from critic using activate_signals to choose modality.
        
        Args:
            critic_observations: Observations for the critic.
                                Shape: (batch_size, num_critic_obs)
                                Structure: [state_goal | human_state | proprioceptive_state]
        
        Returns:
            Value estimates from the critic.
        """
        # Extract proprioceptive state from critic observations
        x_sp_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim:]
        
        # Shared Actor Encoder - encode based on activate_signals (deterministic)
        if self.activate_signals == "robot":
            x_sg_critic = critic_observations[:, :self.actor_sg_dim]
            z = self.actor.encode_robot(x_sg_critic)
        elif self.activate_signals == "smplx":
            x_sh_critic = critic_observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
            z = self.actor.encode_smplx(x_sh_critic)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}")
        
        # Critic MLP: [z, x^p] -> value (Critic_Dual_AE.forward handles normalization)
        value = self.critic(z, x_sp_critic)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_alignment_loss(self, observations):
        """Compute latent space alignment loss (MSE between robot and human encodings).
        
        This loss encourages the robot and human encoders to produce similar
        latent representations for the same input, creating a shared latent space.
        
        Args:
            observations: Input observations containing robot state, human state, and proprioception.
        
        Returns:
            Scalar tensor representing MSE alignment loss.
        """
        # Encode both modalities (deterministic)
        z_robot = self.actor.encode_robot(observations)
        z_human = self.actor.encode_smplx(observations)
        
        # Compute MSE alignment loss
        alignment_loss = torch.mean((z_robot - z_human) ** 2)
        
        return alignment_loss

    def get_reconstruction_loss(self, observations):
        """Compute reconstruction losses for both robot and human decoders.
        
        This loss encourages the decoders to reconstruct input states from their
        corresponding latent representations.
        
        Args:
            observations: Input observations of shape (batch_size, num_critic_obs)
        
        Returns:
            Tuple of (recon_robot_loss, recon_human_loss)
        """
        # Extract state components
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        # Encode both modalities
        z_robot = self.actor.encode_robot(observations)
        z_human = self.actor.encode_smplx(observations)
        
        # Reconstruct and compute losses
        x_sg_recon_robot = self.actor.decode_robot(z_robot)
        recon_robot_loss = torch.mean((x_sg_recon_robot - x_sg) ** 2)
        
        x_sh_recon_human = self.actor.decode_human(z_human)
        recon_human_loss = torch.mean((x_sh_recon_human - x_sh) ** 2)
        
        return recon_robot_loss, recon_human_loss
    
    def get_consistency_loss(self, observations):
        """Compute cross-modal consistency loss for Dual_AE.
        
        This loss ensures consistency between modalities by encoding in one modality
        and decoding with the other modality's decoder, then comparing with the 
        original state of the encoding modality.
        
        When activate_signals == "robot":
            - Encode human state (x_sh) via human_encoder -> z_sh
            - Decode z_sh via robot_decoder -> x_sg_recon_from_human
            - Consistency loss = MSE(x_sg_recon_from_human, x_sg)
            
        When activate_signals == "smplx":
            - Encode robot state (x_sg) via robot_encoder -> z_sg
            - Decode z_sg via human_decoder -> x_sh_recon_from_robot
            - Consistency loss = MSE(x_sh_recon_from_robot, x_sh)
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
                         containing [x_sg | x_sh | x_sp]
        
        Returns:
            consistency_loss: Scalar tensor representing the cross-modal consistency loss
        """
        # Extract states from observations
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        mse_loss = torch.nn.MSELoss()
        
        if self.activate_signals == "robot":
            # Encode human state and decode with robot decoder
            z_sh = self.actor.encode_smplx(x_sh)
            x_sg_recon_from_human = self.actor.decode_robot(z_sh)
            consistency_loss = mse_loss(x_sg_recon_from_human, x_sg)
            
        elif self.activate_signals == "smplx":
            # Encode robot state and decode with human decoder
            z_sg = self.actor.encode_robot(x_sg)
            x_sh_recon_from_robot = self.actor.decode_human(z_sg)
            consistency_loss = mse_loss(x_sh_recon_from_robot, x_sh)
            
        else:
            raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
        
        return consistency_loss

    def get_inference_reconstruction(self, observations):
        """Get reconstruction of states from Dual_AE for analysis.
        
        Performs deterministic reconstruction for both robot and human modalities.
        
        Args:
            observations: Input observations containing robot state, human state, and proprioception.
        
        Returns:
            Tuple of (x_sg, x_sg_recon_robot, x_sh_recon_human)
            where reconstructions come from deterministic encodings (no sampling).
        """
        # Get original states
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        # Encode both modalities (deterministic)
        z_robot = self.actor.encode_robot(observations)
        z_human = self.actor.encode_smplx(observations)
        
        # Reconstruct from latent vectors
        x_sg_recon_robot = self.actor.decode_robot(z_robot)
        x_sh_recon_human = self.actor.decode_human(z_human)
        
        return x_sg, x_sg_recon_robot, x_sh_recon_human
    
    def get_auxiliary_loss(self, observations, loss_items: list[str]=['alignment', 'reconstruction', 'consistency']):
        """Compute all specified losses for Dual_AE actor-critic in one pass.
        
        This method is more efficient than calling individual loss functions separately,
        as it performs encoding only once and computes all requested losses.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
                         containing [x_sg | x_sh | x_sp]
            loss_items: List of loss names to compute. Options:
                        'alignment', 'reconstruction', 'consistency'
                        Default: ['alignment', 'reconstruction', 'consistency']
        
        Returns:
            Dictionary with computed losses. Keys depend on loss_items:
            {
                'alignment': MSE between z_robot and z_human,
                'reconstruction_sg': MSE reconstruction for robot state,
                'reconstruction_sh': MSE reconstruction for human state,
                'consistency': Cross-modal consistency loss,
            }
        """
        # Extract state components once
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        
        # Encode both modalities once (deterministic)
        z_robot = self.actor.encode_robot(observations)
        z_human = self.actor.encode_smplx(observations)
        
        losses = {"alignment": None, "reconstruction_sg": None, "reconstruction_sh": None, "consistency": None}
        mse_loss = torch.nn.MSELoss()
        
        # Compute alignment loss if requested
        if 'alignment' in loss_items:
            alignment_loss = torch.mean((z_robot - z_human) ** 2)
            losses['alignment'] = alignment_loss
        
        # Compute reconstruction losses if requested
        if 'reconstruction' in loss_items:
            # Reconstruct robot state from robot latent
            x_sg_recon_robot = self.actor.decode_robot(z_robot)
            recon_sg_loss = torch.mean((x_sg_recon_robot - x_sg) ** 2)
            losses['reconstruction_sg'] = recon_sg_loss
            
            # Reconstruct human state from human latent
            x_sh_recon_human = self.actor.decode_human(z_human)
            recon_sh_loss = torch.mean((x_sh_recon_human - x_sh) ** 2)
            losses['reconstruction_sh'] = recon_sh_loss
        
        # Compute consistency loss if requested
        if 'consistency' in loss_items:
            if self.activate_signals == "robot":
                # Encode human state and decode with robot decoder
                x_sg_recon_from_human = self.actor.decode_robot(z_human)
                consistency_loss = mse_loss(x_sg_recon_from_human, x_sg)
            elif self.activate_signals == "smplx":
                # Encode robot state and decode with human decoder
                x_sh_recon_from_robot = self.actor.decode_human(z_robot)
                consistency_loss = mse_loss(x_sh_recon_from_robot, x_sh)
            else:
                raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
            losses['consistency'] = consistency_loss
        
        return losses