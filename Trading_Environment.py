# trading environment for Scalping
# Author: Kyle

import pandas as pd
import torch


def load_data():
    tensor = pd.read_csv('dataset.csv', encoding='utf-8')
    tensor = tensor.drop(columns=['Datetime'], axis=1)
    tensor = torch.from_numpy(tensor.values)

    return tensor


class TradingEnvironment():

    def __init__(self, starting_cash, series_length, starting_point):

        tensor = load_data().float()
        self.open_values = pd.DataFrame(tensor[:, 0:1])
        self.high_values = pd.DataFrame(tensor[:, 1:2])
        self.low_values = pd.DataFrame(tensor[:, 2:3])
        self.close_values = pd.DataFrame(tensor[:, 3:4])
        self.volume_values = pd.DataFrame(tensor[:, 5:6])

        self.state = torch.zeros(7, dtype=torch.float).cuda()
        self.starting_cash = starting_cash
        self.starting_point = starting_point
        self.series_length = series_length
        self.current_timestep = starting_point
        self.lose_trades = 0
        self.win_trades = 0

        self.state[0] = self.starting_cash  # cash tracker
        self.state[1] = self.Five_SMA()  # general trend of the past 5 ticks
        self.state[2] = self.volume_values.iloc[self.current_timestep -1][0]  # previous volume traded
        self.state[3] = self.high_values.iloc[self.current_timestep - 1][0]  # previous high value
        self.state[4] = self.low_values.iloc[self.current_timestep - 1][0]  # previous low value
        self.state[5] = self.close_values.iloc[self.current_timestep - 1][0]  # previous close price
        self.state[6] = self.open_values.iloc[self.current_timestep - 1][0]  # previous open price
        # need to use previous time steps cause otherwise we're cheating and seeing the future
        # next open = previous close

        self.done = False

    def portfolio_value(self):
        return self.state[0]

    def update_state(self, cash=None):
        # get all input values for next time step
        if not cash:
            cash = self.state[0]
        SMA = self.Five_SMA()
        Volume = self.volume_values.iloc[self.current_timestep][0]
        current_high = self.high_values.iloc[self.current_timestep][0]
        current_low = self.low_values.iloc[self.current_timestep][0]
        current_close = self.close_values.iloc[self.current_timestep][0]
        current_open = self.open_values.iloc[self.current_timestep][0]
        return cash, SMA, Volume, current_high, current_low, current_close, current_open

    def Five_SMA(self):
        step = self.current_timestep
        if step < 5:
            return self.open_values.iloc[step - 5][0]
        mean = self.open_values.iloc[step - 5:step][0].mean()
        return mean

    '''
    Possible Actions:
    0: buy long scalp
    1: buy short scalp
    2: do nothing/wait
    '''

    def step(self, action):
        cur_timestep = self.current_timestep
        cur_value = self.state[0]
        fee = 5
        # buy_price = (self.high_values.iloc[cur_timestep][0] + self.low_values.iloc[cur_timestep][0])/2
        buy_price = self.open_values.iloc[cur_timestep][0]
        interaction_delta = cur_value - self.portfolio_value()
        gain = 0
        leverage = 100
        nn_modifier = 0
        retval = None

        if self.state[0] < 100:
            nn_modifier = -1000
            self.state = self.update_state()
            return self.state, gain + nn_modifier, True, {"msg": "Bankrupt"}

        if cur_timestep >= self.starting_point + self.series_length:
            self.state = self.update_state()

            if interaction_delta > 250:
                nn_modifier = 250
            return self.state, interaction_delta + nn_modifier - self.lose_trades * 50 + self.win_trades * 100, True, {"msg": "done"}

        if action == 0:
            if buy_price <= (self.high_values.iloc[cur_timestep][0] - 1):
                gain = (self.high_values.iloc[cur_timestep][0] - 1 - buy_price) * leverage - fee
                # gain = 50
                nn_modifier = 250
                self.win_trades += 1

            else:
                gain = -50
                self.lose_trades += 1
                nn_modifier = -50
            self.state = self.update_state(cash=self.state[0] + gain)
            retval = self.state, gain + nn_modifier, False, {"msg": "attempted long scalp"}

        elif action == 1:
            if buy_price >= (self.low_values.iloc[cur_timestep][0] + 1):
                gain = (buy_price - self.low_values.iloc[cur_timestep][0] - 1) * leverage - fee
                # gain = 50
                nn_modifier = 50
                self.win_trades += 1

            else:
                gain = -50
                nn_modifier = -50
                self.lose_trades += 1
            self.state = self.update_state(cash=self.state[0] + gain)
            retval = self.state, gain + nn_modifier, False, {"msg": "attempted short scalp"}

        elif action == 2:
            nn_modifier = 5
            retval = self.update_state(), nn_modifier, False, {"msg": "did nothing"}

        self.current_timestep += 1
        return retval

    def reset(self, starting_point):
        self.state = torch.zeros(7, dtype=torch.float)
        self.starting_point = starting_point
        self.current_timestep = starting_point
        self.lose_trades = 0
        self.win_trades = 0

        self.state[0] = self.starting_cash  # cash tracker
        self.state[1] = self.Five_SMA()  # general trend of the past 5 ticks
        self.state[2] = self.volume_values.iloc[self.current_timestep -1][0]  # previous traded volume
        self.state[3] = self.high_values.iloc[self.current_timestep - 1][0]  # previous high value
        self.state[4] = self.low_values.iloc[self.current_timestep - 1][0]  # previous low value
        self.state[5] = self.close_values.iloc[self.current_timestep - 1][0]  # previous close price
        self.state[6] = self.open_values.iloc[self.current_timestep - 1][0]  # previous open price
        # need to use previous time steps cause otherwise we're cheating and seeing the future

        self.done = False

        return self.state