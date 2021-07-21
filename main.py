import torch
import pandas as pd
import DQN
import matplotlib.pyplot as plt

# data loading testing now that we have csv file to avoid spamming the yfinance api


def main():
    device = torch.device('cuda')
    model = DQN.DQN()
    model.to(device)
    #model.train_model(start_point=0, training_length=2000,
    #                 number_iterations=260, model_name="Roll_index=100", rolling_index=100)
    model.load_state_dict(torch.load("10500_net_profitable_trades=475"))
    model.validate(start=30000, length=5900)
    print("done")


if __name__ == "__main__":
    main()