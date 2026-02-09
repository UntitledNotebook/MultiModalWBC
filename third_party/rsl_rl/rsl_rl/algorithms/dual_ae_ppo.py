import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_dual_ae import ActorCritic_Dual_AE

class Dual_AE_PPO(PPO):
    """
    Dual_AE_PPO extends PPO to integrate Dual Autoencoder for enhanced policy learning.

    This implementation adds multiple loss terms to PPO:
    1. Reconstruction loss: Reconstructs both robot and human states from their respective latent vectors
    2. Alignment loss: Aligns the latent representations of robot and human modalities via MSE
    
    Key differences from VAE_PPO:
    - No reparameterization sampling: uses deterministic encoding
    - No KL divergence loss (Dual_AE doesn't model distributions)
    - No Gaussian alignment loss (replaced with direct latent space MSE alignment)
    - Simpler, more stable training
    
    Attributes:
        reconstruction_loss_coef_sg (float): Coefficient for the robot state reconstruction loss.
        reconstruction_loss_coef_sh (float): Coefficient for the human state reconstruction loss.
        alignment_loss_coef (float): Coefficient for the latent space alignment loss (MSE).
        finetune_human_encoder (bool): Whether to finetune only the human encoder.
    """
    policy: ActorCritic_Dual_AE
    
    def __init__(
        self,
        policy: ActorCritic_Dual_AE,
        reconstruction_loss_coef_sg: float = 0.0,
        reconstruction_loss_coef_sh: float = 0.0,
        alignment_loss_coef: float = 0.0,
        consistency_loss_coef: float = 0.0,
        finetune_human_encoder: bool = False,
        finetune_robot_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.reconstruction_loss_coef_sg = reconstruction_loss_coef_sg
        self.reconstruction_loss_coef_sh = reconstruction_loss_coef_sh
        self.alignment_loss_coef = alignment_loss_coef
        self.consistency_loss_coef = consistency_loss_coef
        
        # Apply finetuning if any networks are specified
        if finetune_human_encoder or finetune_robot_encoder:
            # Determine which networks to finetune
            finetune_networks = []
            if finetune_human_encoder:
                finetune_networks.extend(['human_encoder', 'human_decoder'])
            elif finetune_robot_encoder:
                finetune_networks.extend(['robot_encoder', 'robot_decoder'])
            self.policy.actor.freeze_for_finetune(finetune_networks)


    def update(self):  # noqa: C901
        """
        Perform a PPO update with Dual_AE auxiliary losses, adhering to rollout_storage standards.

        This method computes the standard PPO losses (surrogate loss, value loss, entropy loss)
        and adds Dual_AE-specific losses:
        - Reconstruction loss: Robot decoder reconstructs robot state; human decoder reconstructs human state
        - Alignment loss: MSE between robot and human latent representations
        """
        # Initialize mean losses for logging
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_recon_sg_loss = 0
        mean_recon_sh_loss = 0
        mean_alignment_loss = 0
        mean_consistency_loss = 0

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None

        # Use non-recurrent mini-batch generator (drop is_recurrent branches)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)


            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean())

            ############################################################
            #          DUAL_AE AUXILIARY LOSSES                        #
            ############################################################         
            loss_dict = self.policy.get_auxiliary_loss(obs_batch)
            
            recon_sg_loss = loss_dict['reconstruction_sg']
            recon_sh_loss = loss_dict['reconstruction_sh']
            alignment_loss = loss_dict['alignment']
            consistency_loss = loss_dict['consistency']
            
            # Total loss: PPO loss + Dual_AE auxiliary losses
            loss += (
                self.reconstruction_loss_coef_sg * recon_sg_loss
                + self.reconstruction_loss_coef_sh * recon_sh_loss
                + self.alignment_loss_coef * alignment_loss
                + self.consistency_loss_coef * consistency_loss
            )

            ############################################################
            #          END DUAL_AE AUXILIARY LOSSES                    #
            ############################################################

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_recon_sg_loss += recon_sg_loss.item()
            mean_recon_sh_loss += recon_sh_loss.item()
            mean_alignment_loss += alignment_loss.item()
            mean_consistency_loss += consistency_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_recon_sg_loss /= num_updates
        mean_recon_sh_loss /= num_updates
        mean_alignment_loss /= num_updates
        mean_consistency_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "reconstruction_sg": mean_recon_sg_loss,
            "reconstruction_sh": mean_recon_sh_loss,
            "alignment": mean_alignment_loss,
            "consistency": mean_consistency_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        return loss_dict
