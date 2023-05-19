import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from ucb import UCB
from ts import TS
from etc import ETC
import os

def plot_regret(ucb_results, ts_results, etc_results, T, alpha, ax):
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
    ax.plot(rounds, ucb_mean, label='UCB')
    ax.fill_between(rounds, ucb_mean - 2*ucb_std, ucb_mean + 2*ucb_std, alpha=0.2)

    ax.plot(rounds, ts_mean, label='TS')
    ax.fill_between(rounds, ts_mean - 2*ts_std, ts_mean + 2*ts_std, alpha=0.2)
    
    ax.plot(rounds, etc_mean, label='ETC')
    ax.fill_between(rounds, etc_mean - 2*etc_std, etc_mean + 2*etc_std, alpha=0.2)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Cumulative Regret')
    ax.legend()
    ax.set_title(f'alpha = {alpha}')

K = 10
T = 10000
mu = np.random.rand(K)
c = np.random.rand(K)
num_trials = 50
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]  # Add or remove alphas as needed

# Create a new figure
fig, axs = plt.subplots(len(alphas), figsize=(10, 5*len(alphas)))

for index, alpha in enumerate(alphas):
    ucb_results = []
    ts_results = []
    etc_results = []

    for trial in range(num_trials):
        np.random.seed(trial)

        bandit = Bandit(K, mu, c, alpha)
        ucb = UCB(bandit, T)
        ts = TS(bandit, T)
        etc = ETC(bandit, T, tau=10)

        ucb_results.append(ucb.run())
        ts_results.append(ts.run())
        etc_results.append(etc.run())

    # Plot the results for this alpha
    plot_regret(ucb_results, ts_results, etc_results, T, alpha, axs[index])

# Save the figure to results directory
os.makedirs('results', exist_ok=True)
plt.savefig('results/MAB_results.png')
