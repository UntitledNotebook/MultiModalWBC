import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_vae import ActorCriticVAE

class VAE_PPO(PPO):
    """
    VAE_PPO extends PPO to integrate VAE for enhanced policy learning.

    This implementation adds multiple loss terms to PPO:
    1. Reconstruction loss: Reconstructs both robot and human encodings to robot command state
    2. Gaussian alignment loss: Aligns the mean and log-variance of robot and human encodings
    3. KL divergence loss: Encourages the latent distribution to match a standard normal
    
    Attributes:
        reconstruction_loss_coef_sg (float): Coefficient for the robot state reconstruction loss.
        reconstruction_loss_coef_sh (float): Coefficient for the human state reconstruction loss.
        gaussian_alignment_loss_coef (float): Coefficient for the Gaussian distribution alignment loss.
        kl_loss_coef (float): Coefficient for the KL divergence loss.
        finetune_human_encoder (bool): Whether to finetune only the human encoder.
    """
    policy: ActorCriticVAE
    
    def __init__(
        self,
        policy: ActorCriticVAE,
        reconstruction_loss_coef_sg: float = 1.0,
        reconstruction_loss_coef_sh: float = 1.0,
        gaussian_alignment_loss_coef: float = 1.0,
        kl_loss_coef: float = 1.0,
        finetune_human_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.reconstruction_loss_coef_sg = reconstruction_loss_coef_sg
        self.reconstruction_loss_coef_sh = reconstruction_loss_coef_sh
        self.gaussian_alignment_loss_coef = gaussian_alignment_loss_coef
        self.kl_loss_coef = kl_loss_coef
        
        self.finetune_human_encoder = finetune_human_encoder
        if self.finetune_human_encoder:
            print("[INFO] VAE_PPO: Finetuning human encoder only.")
            self.policy.actor.freeze_for_finetune()

    def update(self):  # noqa: C901
        """
        Perform a PPO update with VAE auxiliary losses, adhering to rollout_storage standards.

        This method computes the standard PPO losses (surrogate loss, value loss, entropy loss)
        and adds VAE-specific losses:
        - Reconstruction loss: Both robot and human modalities reconstruct to robot goal state
        - Gaussian alignment loss: Aligns mean and log-variance between robot and human encoders
        - KL divergence loss: Encourages latent distributions to match a standard normal
        """
        # Initialize mean losses for logging
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_recon_sg_loss = 0
        mean_recon_sh_loss = 0
        mean_gaussian_align_loss = 0
        mean_kl_loss = 0

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
            #              VAE AUXILIARY LOSSES                        #
            ############################################################
            # Get reconstruction and latent parameters from both modalities
            x_sg, recon_sg_robot, recon_sg_human = self.policy.get_inference_reconstruction(obs_batch)
            
            # Reconstruction loss: both modalities reconstruct to robot goal state
            recon_sg_loss = nn.functional.mse_loss(x_sg, recon_sg_robot)  # robot reconstruction
            recon_sh_loss = nn.functional.mse_loss(x_sg, recon_sg_human)  # human aligns to robot goal state
            
            # Encode with sampling to get latent parameters for alignment loss
            z_robot, mu_robot, logvar_robot = self.policy.actor.encode_robot(obs_batch)
            z_human, mu_human, logvar_human = self.policy.actor.encode_smplx(obs_batch)
            
            # Gaussian alignment loss: KL divergence between robot and human encoder distributions
            # KL(N(mu_robot, sigma_robot) || N(mu_human, sigma_human))
            # = 0.5 * (logvar_human - logvar_robot - 1 + (sigma_robot^2 + (mu_robot - mu_human)^2) / sigma_human^2)
            var_robot = logvar_robot.exp()
            var_human = logvar_human.exp()
            gaussian_align_loss = 0.5 * torch.mean(
                logvar_human - logvar_robot 
                - 1 
                + var_robot / var_human 
                + (mu_robot - mu_human).pow(2) / var_human
            )
            
            # KL divergence loss: encourage latent distribution to match N(0, I)
            # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_robot = -0.5 * torch.sum(1 + logvar_robot - mu_robot.pow(2) - logvar_robot.exp(), dim=-1).mean()
            kl_human = -0.5 * torch.sum(1 + logvar_human - mu_human.pow(2) - logvar_human.exp(), dim=-1).mean()
            kl_loss = kl_robot + kl_human
            
            # Total loss: PPO loss + VAE auxiliary losses
            loss += (
                self.reconstruction_loss_coef_sg * recon_sg_loss
                # + self.reconstruction_loss_coef_sh * recon_sh_loss
                # + self.gaussian_alignment_loss_coef * gaussian_align_loss
                + self.kl_loss_coef * kl_loss
            )

            ############################################################
            #              END VAE AUXILIARY LOSSES                    #
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
            mean_gaussian_align_loss += gaussian_align_loss.item()
            mean_kl_loss += kl_loss.item()
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
        mean_gaussian_align_loss /= num_updates
        mean_kl_loss /= num_updates
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
            "gaussian_alignment": mean_gaussian_align_loss,
            "kl_divergence": mean_kl_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        return loss_dict