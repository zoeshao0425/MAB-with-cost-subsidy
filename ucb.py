import numpy as np

class UCB:
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.T = T
        self.mu_score = np.zeros(bandit.K)
        self.T_i = np.zeros(bandit.K)
        self.rewards = np.zeros((T, bandit.K))
        self.costs = np.zeros(T)

    def update_score(self, t, i, reward):
        self.mu_score[i] = np.sum(self.rewards[:t, i]) / self.T_i[i]
        beta = np.sqrt((2 * np.log(self.T)) / self.T_i[i])
        self.mu_score[i] = min(self.mu_score[i] + beta, 1)

    def run(self):
        for t in range(self.T):
            if t < self.bandit.K:  # play each arm once
                i = t
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.rewards[t, i] = reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]
            else:
                for i in range(self.bandit.K):
                    self.update_score(t, i, self.rewards[t-1, i])
                m_t = np.argmax(self.mu_score)
                feasible_set = [i for i in range(self.bandit.K) if self.mu_score[i] - (1 - self.bandit.alpha) * self.mu_score[m_t] >= 0]
                i = min(feasible_set, key=lambda x: self.bandit.c[x])
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.rewards[t, i] = reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]

        return np.cumsum(self.costs)