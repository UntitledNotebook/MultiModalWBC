# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

from .actor_critic_vae import ActorCriticVAE
from .actor_critic_fsqvae import ActorCriticFSQVAE
from .actor_critic_sonic import ActorCriticSONIC
from .actor_critic_projection import ActorCriticProjection
from .actor_critic_dual_ae import ActorCritic_Dual_AE
from .actor_critic_triple_ae import ActorCritic_Triple_AE, ActorCritic_Triple_AE_Single_Finetune

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",

    "ActorCriticVAE",
    "ActorCriticFSQVAE",
    "ActorCriticSONIC",
    "ActorCriticProjection",
    "ActorCritic_Dual_AE",
    "ActorCritic_Triple_AE",
    "ActorCritic_Triple_AE_Single_Finetune",
]
