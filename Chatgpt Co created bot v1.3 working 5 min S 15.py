#!pip install yfinance
#!pip install stable-baselines3
#!pip install alpaca-trade-api
#!pip install ta
#!pip install schedule
#!pip install 'shimmy>=0.2.1

import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from ta import add_all_ta_features
import alpaca_trade_api as tradeapi
from datetime import datetime
import schedule
import time
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
sys.path.append('C:/Users/Milan/OneDrive/Documents/Chat gpt3/Github alapcahq test')
print(sys.path)

# Import the API keys from config.py
import key_config 

# Set Alpaca API key and secret
API_KEY = key_config.APCA_API_KEY_ID
API_SECRET = key_config.APCA_API_SECRET_KEY
BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading URL

stock_symbol = "SPY"  # Change to your desired stock ticker
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)


#Data Preparation:

def fetch_data2():
    print("getting data")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=55)
    # Ensure the granularity is adjusted to your preference. Here's an example for 5-minute intervals
    data = yf.download(stock_symbol, start=start_date, end=end_date, interval="5m")
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return data[['Close', 'volume_adi', 'volatility_kcw', 'trend_ema_fast', 'momentum_uo', 'others_dr']].values

def fetch_data1():
    print("getting data")
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start="2020-01-01", end=end_date)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return data[['Close', 'volume_adi', 'volatility_kcw', 'trend_ema_fast', 'momentum_uo', 'others_dr']].values

data = fetch_data1()

# trading Envir
class TradingEnv(gym.Env):
    print(" In the Gym, env")
    def __init__(self, data, api, stock_symbol):
        super(TradingEnv, self).__init__()
        self.data = data
        self.api = api  # Alpaca API
        self.stock_symbol = stock_symbol
        self.current_step = 0
        
        # Initialize account and stock details
        self.balance = float(self.api.get_account().cash)
        print(f"self balance {self.balance}  ")
        try:
            position = self.api.get_position(self.stock_symbol)
            self.shares_held = int(position.qty)
            print(f"Shares of {position} qty is {self.shares_held}  ")
        except:
            self.shares_held = 0  # No position exists
           

        # Define action and observation spaces
        # This is just an example; you may need to adjust the spaces according to your needs
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold as an example
        self.observation_space = spaces.Box(low=0, high=1, 
                                        shape=(len(self.data[self.current_step]) + 2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        obs = self._next_observation()  # This will initialize the self.share_price
        self.start_of_day_value = self.balance + (self.shares_held * self.share_price)
        return obs

    def _next_observation(self):
        # Extract the market data
        obs = self.data[self.current_step].copy()
    
        # Assuming the first column in your data is the close price
        self.share_price = obs[0]  
    
        # Append balance, shares_held, and share_price
        obs = np.append(obs, [self.balance, self.shares_held])
    
        return obs

    def step(self, action):
        # Save the previous total asset value
        prev_total_value = self.balance + (self.shares_held * self.share_price)

        # Handle buy/sell/hold operations
        # If the action is to buy
        if action == 0:
            # Calculate the number of shares you can buy with the current balance
            shares_to_buy = self.balance // self.share_price
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * self.share_price
                self.shares_held += shares_to_buy
    
        # If the action is to sell
        elif action == 1:
            if self.shares_held > 0:
                self.balance += self.shares_held * self.share_price
                self.shares_held = 0

        # For action == 2, just hold, so no changes to balance or shares_held

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        # Calculate the new total asset value
        current_total_value = self.balance + (self.shares_held * self.share_price)
    
        # Reward is the difference in total asset values
        reward = current_total_value - prev_total_value
        

        # Compute daily return if at the end of a day or episode
        if self.current_step % 78 == 0 or done:
            daily_return = current_total_value - self.start_of_day_value
            reward += daily_return  # Combine both rewards, or you can handle it differently based on your requirements
            self.start_of_day_value = current_total_value  # Reset for the next day


        return self._next_observation(), reward, done, {}



    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass



#Environment Initialization:
env = TradingEnv(data, api, stock_symbol)
env = DummyVecEnv([lambda: env])  # Wrap it for stable-baselines3
print("initi Envir")

#save ppo
model_path = "ppo_trading_model"

# Check if the model file exists
print("Checking if model exists...")
if os.path.isfile(model_path + ".zip"):
    model = PPO.load(model_path, env=env)
    model.learn(total_timesteps=10000)  # Continue training
    model.save(model_path)  # Save the re-trained model

    print(f"Model loaded from {model_path}")
else:
    #PPO Agent Training:
    print("Starting model training since saved model not found...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def train_model():
    # Check if the model file exists
    print("Checking if model exists...")
    if os.path.isfile(model_path + ".zip"):
        model = PPO.load(model_path, env=env)
        model.learn(total_timesteps=10000)  # Continue training
        model.save(model_path)  # Save the re-trained model
        print(f"Model loaded and trained from {model_path}")
    else:
        # PPO Agent Training:
        print("Starting model training since saved model not found...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(model_path)
        print(f"Model saved to {model_path}")



print(" saved and pull model")



#Trading Logic


def trade():
    print("trading")
    clock = api.get_clock()
    if not clock.is_open:
        print("Market is not open. Exiting.")
        return

    obs = env.reset()
    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # Get current position for the stock
        try:
            position = api.get_position(stock_symbol)
            current_qty = int(position.qty)
        except:
            current_qty = 0  # No position exists
        
        if action == 0 and current_qty >= 0:  # Assuming 0 is Buy
            api.submit_order(
                symbol=stock_symbol,
                qty=1,  # Adjust quantity as needed
                side="buy",
                type="market",
                time_in_force="gtc"
            )
        elif action == 1 and current_qty > 0:  # Assuming 1 is Sell and we have stocks to sell
            api.submit_order(
                symbol=stock_symbol,
                qty=1,  # Adjust quantity as needed
                side="sell",
                type="market",
                time_in_force="gtc"
            )
        # if action == 2, then it's "hold", so do nothing.

        if done:
            break




def update_data1():
    global data
    global env
    data = fetch_data1()
    print(" pulling data 1")
    # Reset the environment with updated data
    env = TradingEnv(data, api, stock_symbol)
    env = DummyVecEnv([lambda: env])

def update_data2():
    global data
    global env
    data = fetch_data2()
    print(" pulling data 2")
    # Reset the environment with updated data
    env = TradingEnv(data, api, stock_symbol)
    env = DummyVecEnv([lambda: env])


# Schedule the data update every 15 minutes
schedule.every(60).minutes.do(update_data1)
schedule.every(2).minutes.do(update_data2)
#schedule.every(3).minutes.do(train_model)  # Retrain model after updating data
schedule.every().day.at("16:30").do(train_model)  # Retrain model after updating data
schedule.every().day.at("08:30").do(trade)
while True:
    
    schedule.run_pending()
    time.sleep(1)