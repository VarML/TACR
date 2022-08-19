import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tac.training.trainer import Trainer

class SequenceTrainer(Trainer):

    def train_step(self):
        # Algorithm 1, line6 : Sample a random minibatch
        states, actions, rewards, dones, next_state, next_actions, next_rewards, \
        timesteps, next_timesteps, attention_mask = self.get_batch(self.batch_size)

        # # Algorithm 1, line9 : Predict a action
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )

        next_state_preds, next_action_preds, next_reward_preds = self.actor_target.forward(
            next_state, next_actions, next_rewards, next_timesteps, attention_mask=attention_mask,
        )
        
        states = states.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        next_state = next_state.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        action_sample = actions.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        Q_action_preds = action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        next_Q_action_preds = next_action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

        # Algorithm 1, line10, line11
        # Compute the target Q value
        with torch.no_grad():
            target_Q1 = self.critic_target(next_state, next_Q_action_preds)
            target_Q = torch.min(target_Q1)
            target_Q = rewards + dones * self.discount * target_Q
        # Get current Q estimates
        current_Q = self.critic(states, action_sample)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Algorithm 1, line12, line13 : Set lambda and Compute actor loss
        pi = Q_action_preds
        Q = self.critic.Q1(states, pi)
        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action_sample)

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .25)      
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        # # Algorithm 1, line16 : Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau  * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.detach().cpu().item()
    
        