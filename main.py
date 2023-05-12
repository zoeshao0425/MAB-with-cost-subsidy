import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import random
from bandit import Bandit
from ucb import UCB
from ts import TS
from etc import ETC

K = 10
T = 10000
mu = np.random.rand(K)
c = np.random.rand(K)
alpha = 0.1

def plot_regret(ucb_results, ts_results, etc_results, T):
    # Convert the lists of results into numpy arrays for easier calculations
    ucb_results = np.array(ucb_results)
    ts_results = np.array(ts_results)
    etc_results = np.array(etc_results)

    # Calculate the mean and standard deviation of the results for each round
    ucb_mean = ucb_results.mean(axis=0)
    ts_mean = ts_results.mean(axis=0)
    etc_mean = etc_results.mean(axis=0)
    ucb_std = ucb_results.std(axis=0)
    ts_std = ts_results.std(axis=0)
    etc_std = etc_results.std(axis=0)

    # Generate the x values
    rounds = np.arange(T)

    # Plot the mean results with error bars representing two standard deviations
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, ucb_mean, label='UCB')
    plt.fill_between(rounds, ucb_mean - 2*ucb_std, ucb_mean + 2*ucb_std, alpha=0.2)

    plt.plot(rounds, ts_mean, label='TS')
    plt.fill_between(rounds, ts_mean - 2*ts_std, ts_mean + 2*ts_std, alpha=0.2)
    
    plt.plot(rounds, etc_mean, label='ETC')
    plt.fill_between(rounds, etc_mean - 2*etc_std, etc_mean + 2*etc_std, alpha=0.2)

    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.title('Cumulative Regret of UCB, TS and UCB Over Time')
    plt.show()

ucb_results = []
ts_results = []
etc_results = []
for trial in range(50):
    np.random.seed(trial*2)
    
    bandit = Bandit(K, mu, c, alpha)
    ucb = UCB(bandit, T)
    ts = TS(bandit, T)
    etc = ETC(bandit, T, tau=10)

    ucb_results.append(ucb.run())
    ts_results.append(ts.run())
    etc_results.append(etc.run())

# Plot the results
plot_regret(ucb_results, ts_results, etc_results, T)
