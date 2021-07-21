import torch
import torch.nn as nn
import Trading_Environment as ENV

from collections import namedtuple
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# everything runs from the main file now



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(DQN, self).__init__()

        self.fcl1 = nn.Linear(7, 128)
        self.fcl2 = nn.Linear(128, 128)
        self.fcl3 = nn.Linear(32, 32)
        self.hidden_state1 = torch.tensor(torch.zeros(2, 1, 32), requires_grad=False).cuda()
        self.rnn1 = nn.GRU(128, 32, 2)
        self.fcl4 = nn.Linear(32, 5)
        self.fcl5 = nn.Linear(5, 128)
        self.fcl6 = nn.Linear(32, 31)
        self.hidden_state2 = torch.tensor(torch.zeros(2, 1, 32), requires_grad=False).cuda()
        self.rnn2 = nn.GRU(128, 32, 2)
        self.action_head = nn.Linear(31, 3)
        self.value_head = nn.Linear(31, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = torch.tensor(x).cuda()
        x = torch.sigmoid(self.fcl1(x))
        x = torch.tanh(self.fcl2(x))
        x, self.hidden_state1 = self.rnn1(x.view(1, -1, 128), self.hidden_state1.data)
        x = F.relu(self.fcl3(x.squeeze()))
        x = torch.sigmoid(self.fcl4(x))
        x = torch.tanh(self.fcl5(x))
        x, self.hidden_state2 = self.rnn2(x.view(1, -1, 128), self.hidden_state2.data)
        x = F.relu(self.fcl6(x.squeeze()))
        action_scores = self.action_head(x.squeeze())
        state_values = self.value_head(x.squeeze())
        return F.softmax(action_scores, dim=-1), state_values

    def act(self, state):
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append([m.log_prob(action), state_value])
        return action.item()

    def train_model(self, start_point, training_length, number_iterations,
                    model_name, start_cash=2000, rolling_index=0):
        gamma = 0.9
        log_interval = 20
        running_reward = 0
        model = self
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        env = ENV.TradingEnvironment(starting_cash=start_cash, starting_point=start_point, series_length=training_length)

        for episode in range(0, number_iterations):
            R = 0
            saved_actions = model.saved_actions
            reward = 0
            policy_losses = []
            value_losses = []
            rewards = []
            done = False
            msg = None
            state = env.reset(env.starting_point+rolling_index)

            while not done:  # do a run through then back prop rewards once the run is finished
                action = model.act(state)
                state, reward, done, msg = env.step(action)
                model.rewards.append(reward)
                if done:
                    break
            running_reward = running_reward * (1 - 1 / log_interval) + reward * (1 / log_interval)
            for r in model.rewards[::-1]:
                R = r + (gamma * R)
                rewards.insert(0, R)
            rewards = torch.tensor(rewards)

            epsilon = (torch.rand(1) / 1e4) - 5e-4
            rewards += epsilon

            for (log_prob, value), r in zip(saved_actions, rewards):
                reward = torch.tensor(r - value.item()).cuda()
                policy_losses.append(-log_prob * reward)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).cuda()))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            loss = torch.clamp(loss, -1e-5, 1e5)
            loss.backward()
            optimizer.step()
            del model.rewards[:]
            del model.saved_actions[:]

            gain = env.portfolio_value() - env.starting_cash
            net_trades = env.win_trades - env.lose_trades
            if msg["msg"] == "done" and (net_trades >= 4) and gain > 8000:
                print("""Success_save: Reward = {}"  + " final portfolio Val: {},Timestep: {}, net profit: {}, message: {}"""
                      .format(int(reward),env.portfolio_value(), env.current_timestep, gain, msg["msg"]))
                highest_gain = gain
                torch.save(model.state_dict(), '{}_net_profitable_trades={}'.format(env.current_timestep, net_trades))
            elif msg["msg"] == "done":
                print("""Finished full training cycle: net profit = {}, number of profitable trades = {},
                number of losing trades = {}, episode = {}""".format(gain,env.win_trades,env.lose_trades, episode))
                new_model_name = "{}.WLR.{}.{}.{}".format(model_name,env.win_trades,env.lose_trades, env.portfolio_value())
                torch.save(model.state_dict(), new_model_name)
            if episode % log_interval == 0:
                print("""Episode {}: started at {:.1f}, finished at {:.1f} because {} @ t={}, \
        last reward {:.1f}, running reward {:.1f}, Win trades: {}, Lose trades: {}""".format(episode, env.starting_cash,
                                                            env.portfolio_value(), msg["msg"], env.current_timestep,
                                                            reward, running_reward, env.win_trades, env.lose_trades))

    def validate(self, cash=2000, start=24000, length=6000):
        model = self
        model.eval()
        torch.no_grad()
        env = ENV.TradingEnvironment(starting_point=start, starting_cash=cash, series_length=length)
        gain = env.portfolio_value() - env.starting_cash

        for i in range(0, length):
            action = model.act(env.state)
            next_state, reward, done, msg = env.step(action)
            print(msg, """ win trades = {}, lose trades = {},cash = {}, time = {}""".
                  format(env.win_trades,env.lose_trades, env.portfolio_value(), env.current_timestep))

            if msg["msg"] == 'Bankrupt':
                print('bankrupted self at t = {}'.format(env.current_timestep))
                break
            if msg["msg"] == "done":
                print("""Finished full training cycle: net profit = {}, number of profitable trades = {},
                                number of losing trades = {}""".format(gain, env.win_trades, env.lose_trades))













