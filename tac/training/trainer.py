import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        
        return q

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q= F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, state_dim, action_dim, K, loss_fn=None,scheduler=None, eval_fns=None):
        self.actor = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K
        self.total_it = 0
        
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-6)

        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 4.4

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.actor.train()
        for _ in range(num_steps):
            self.total_it += 1
            train_loss = self.train_step()
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.actor.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

