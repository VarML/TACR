import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import os
matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """Stock trading environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        initial_amount : int
            start money
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date
    Methods
    -------
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        initial_amount,
        state_space,
        action_space,
        tech_indicator_list,
        dataset=None,
        turbulence_threshold=None,
        mode="",
        lookback=252,
        day=0,
    ):
        self.dataset=dataset
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.mode=mode
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    +self.data.close.values.tolist()
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount
        self.turbulence=0
        self.pre_weights=0
        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            if not os.path.exists("results"):
                os.makedirs("results")

            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/"+self.dataset+"_cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/"+self.dataset+"_rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(int(self.portfolio_value)))
            

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]

            # Equation (16) : Compute Sharpe ratio
            df_daily_return[1:]-=0.02/365 #bank interest
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std())

                print("Sharpe: ", round(sharpe,3),)
            print("=================================")

            if self.mode=="test":
                # df_actions = self.save_action_memory()
                # df_actions.to_csv(
                #     "results/actions_{}.csv".format(
                #         self.mode
                #     )
                # )


                df_asset = self.save_asset_memory()
                df_asset.to_csv(
                    "results/{}_asset_{}.csv".format(
                        self.dataset,
                        self.mode
                    )
                )
                
                plt.plot(df_asset, "r")
                # plt.savefig(
                #     "results/{}_account_value_{}.png".format(
                #         self.dataset,
                #         self.mode
                #     ),
                #     index=False,
                # )
                # plt.close()

            return self.state, self.reward, self.terminal, {}

        else:
            weights = self.softmax_normalization(actions)

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    weights=np.zeros(len(weights),dtype=float)

            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data["turbulence"]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data["turbulence"].values[0]

            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )

            # print(self.state)
            # calcualte portfolio return
            # Equation (1), (2) : individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)

            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = portfolio_return
        
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence=0
        # load states
        self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )

        
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions
    
    def save_asset_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        asset_list = self.asset_memory
        df_asset = pd.DataFrame(asset_list)
        df_asset.columns = ["asset"]
        df_asset.index = df_date.date
        return df_asset

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]