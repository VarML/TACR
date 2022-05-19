import time
import torch
import argparse
import pandas as pd
import numpy as np
import pickle
from finrl.apps import config
from finrl.neo_finrl.env_portfolio_allocation.env_portfolio2 import StockPortfolioEnv
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg,evaluate,eval_test
from decision_transformer.models.decision_transformer import DecisionTransformer


def experiment(exp_prefix, variant,):
    device = variant.get('device', 'cuda')

    train = pd.read_csv("sp/sp_train.csv", index_col=[0])
    trade = pd.read_csv("sp/sp_trade.csv", index_col=[0])
    max_ep_len = train.index[-1]

    stock_dimension = len(trade.tic.unique())
    #state_space = 4 * stock_dimension
    state_space = 4*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.0025,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "mode":"test"
    }
    env = StockPortfolioEnv(df=trade, **env_kwargs)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    trajectories = []
    for i in range(5):
        dataset_path = f'sp/{"stock" + str(i + 1)}-{"train"}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            tra = pickle.load(f)
        trajectories.append(tra[0])

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        path['rewards'][-1] = path['rewards'].sum()
        path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    K = variant['K']

    model = DecisionTransformer(
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
        attn_pdrop=variant['dropout'])

    model.load_state_dict(torch.load('asp40.pt'))

    eval_test(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    x=time.time()
    experiment('gym-experiment', variant=vars(args))
    print("learning time",(time.time()-x)/3600)