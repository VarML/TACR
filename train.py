import numpy as np
import wandb
import argparse
import random
import pickle
import pandas as pd
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
import torch
from tac.models.transformer_actor import TransformerActor
from tac.training.seq_trainer import SequenceTrainer

def experiment(variant):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    train = pd.read_csv("datasets/train.csv", index_col=[0])
    max_ep_len = train.index[-1]

    stock_dimension = len(train.tic.unique())
    state_space = 4*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "initial_amount": 1000000,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
    }
    env = StockPortfolioEnv(df=train, **env_kwargs)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    K = variant['K']

    model = TransformerActor(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )

    trajectories=[]
    for i in range(5):
        dataset_path = f'trajectory/{"stock"+str(i+1)}-{"train"}.pkl'
        with open(dataset_path, 'rb') as f:
            tra = pickle.load(f)
        trajectories.append(tra[0])

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest

    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2

    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        s, next_s, next_a, next_r, a, r, d, dd, timesteps, n_timesteps, mask \
            = [], [], [], [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            dd.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))

            if si >= traj['rewards'].shape[0] - K:
                next_s.append(np.append(traj['observations'][si + 1:si + 1 + max_len],
                                        traj['observations'][traj['rewards'].shape[0] - 1]).reshape(1, -1, state_dim))
                next_a.append(np.append(traj['actions'][si + 1:si + 1 + max_len],
                                        traj['actions'][traj['rewards'].shape[0] - 1]).reshape(1, -1, act_dim))
                next_r.append(np.append(traj['rewards'][si + 1:si + 1 + max_len],
                                        np.array([traj['rewards'][traj['rewards'].shape[0] - 1]])).reshape(1, -1, 1))
            else:
                next_s.append(traj['observations'][si + 1:si + 1 + max_len].reshape(1, -1, state_dim))
                next_a.append(traj['actions'][si + 1:si + 1 + max_len].reshape(1, -1, act_dim))
                next_r.append(traj['rewards'][si + 1:si + 1 + max_len].reshape(1, -1, 1))

            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            n_timesteps.append(np.arange(si + 1, si + 1 + s[-1].shape[1]).reshape(1, -1))
            n_timesteps[-1][n_timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            # padding and state + reward normalization

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            next_s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), next_s[-1]], axis=1)
            next_s[-1] = (next_s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            dd[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), dd[-1]], axis=1)
            next_a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., next_a[-1]], axis=1)
            next_r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), next_r[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            n_timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), n_timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        next_s = torch.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        dd = torch.from_numpy(np.concatenate(dd, axis=0)).to(dtype=torch.float32, device=device)
        next_a = torch.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=torch.float32, device=device)
        next_r = torch.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        n_timesteps = torch.from_numpy(np.concatenate(n_timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, dd, next_s, next_a, next_r, timesteps, n_timesteps, mask


    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        action_dim=act_dim,
        state_dim=state_dim,
        K=K
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='tac',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

    torch.save(trainer.actor.state_dict(), group_name+'.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--dataset', type=str, default='dow')
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=4000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()
    experiment(variant=vars(args))
