from preprocessor.process_traj import trajectory
import pandas as pd
from stock_env.apps import config
import pickle
import numpy as np

train = pd.read_csv("datasets/train.csv", index_col=[0])

stock_dimension = len(train.tic.unique())
state_space = 4 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension
    }
env = trajectory(df=train, **env_kwargs)

def traj_generator(env, episode):
    ob = env.reset()
    obs = []
    rews = []
    term = []
    acs = []

    while True:
        next_state, rew, new, ac = env.step(episode)
        obs.append(ob)
        term.append(new)
        acs.append(ac)
        rews.append(rew)
        ob = next_state

        if new:
            break

    obs = np.array(obs)
    rews = np.array(rews)
    term = np.array(term)
    acs = np.array(acs)

    traj = {"observations": obs, "rewards": rews, "dones": term, "actions": acs}
    return traj

for i in range(5):
    traj = traj_generator(env, i)

    episode_data = {}
    paths = []

    for k in traj:
        episode_data[k] = np.array(traj[k])
    paths.append(episode_data)
    name = f'{"trajectory/stock"+str(i+1)}-{"train"}-v2'
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)