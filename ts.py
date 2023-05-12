import numpy as np


class TS:
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.T = T
        self.mu_score = np.zeros(bandit.K)
        self.T_i = np.zeros(bandit.K)
        self.successes = np.zeros(bandit.K)
        self.failures = np.zeros(bandit.K)
        self.costs = np.zeros(T)

    def update_score(self, i):
        # Sample from the posterior distribution (Beta) of each arm
        self.mu_score[i] = np.random.beta(self.successes[i] + 1, self.failures[i] + 1)

    def run(self):
        for t in range(self.T):
            if t < self.bandit.K:  # play each arm once
                i = t
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.successes[i] += reward
                self.failures[i] += 1 - reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]
            else:
                for i in range(self.bandit.K):
                    self.update_score(i)
                m_t = np.argmax(self.mu_score)
                feasible_set = [i for i in range(self.bandit.K) if self.mu_score[i] - (1 - self.bandit.alpha) * self.mu_score[m_t] >= 0]
                i = min(feasible_set, key=lambda x: self.bandit.c[x])
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.successes[i] += reward
                self.failures[i] += 1 - reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]

        return np.cumsum(self.costs)