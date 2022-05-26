import numpy as np
import pandas as pd

class trajectory:

    def __init__(
        self,
        df,
        stock_dim,
        state_space,
        action_space,
        tech_indicator_list,
        day=0,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        
        self.data = self.df.loc[self.day, :]
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
        self.terminal = False

    def step(self,i):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            return self.state, self.reward, self.terminal, np.zeros(self.action_space,dtype=float)

        else:
            last_day_memory = self.data
            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
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
            
            portion = (self.data.close.values / last_day_memory.close.values)
            bc = []

            if i==4:
                for j in portion:
                    bc.append(4**(j*3))
            else:
                for j in portion:
                    bc.append(3**(j*(i+1)))
                    
            weights = self.softmax_normalization(bc)
            weights[np.isnan(weights)] = 1.

            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )

            self.reward = portfolio_return

        return self.state, self.reward, self.terminal, weights

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
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
                    #+self.data.turbulence.tolist()

                )
        self.terminal = False
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output