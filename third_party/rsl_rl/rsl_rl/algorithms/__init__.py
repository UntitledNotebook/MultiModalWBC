# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO
from .vae_ppo import VAE_PPO
from .fsqvae_ppo import FSQVAE_PPO
from .sonic_ppo import SONIC_PPO
from .projection_ppo import Projection_PPO
from .dual_ae_ppo import Dual_AE_PPO
from .triple_ae_ppo import Triple_AE_PPO, Triple_AE_PPO_Single_Finetune

__all__ = ["PPO", "Distillation", "VAE_PPO", "FSQVAE_PPO", "SONIC_PPO", "Projection_PPO", "Dual_AE_PPO", "Triple_AE_PPO", "Triple_AE_PPO_Single_Finetune"]