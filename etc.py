import numpy as np

class ETC:
    def __init__(self, bandit, T, tau):
        self.bandit = bandit
        self.T = T
        self.tau = tau
        self.mu_hat = np.zeros(bandit.K)
        self.mu_ucb = np.zeros(bandit.K)
        self.mu_lcb = np.zeros(bandit.K)
        self.T_i = np.zeros(bandit.K)
        self.rewards = np.zeros((bandit.K, tau))
        self.costs = np.zeros(T)

    def update_score(self, t, i):
        self.mu_hat[i] = np.sum(self.rewards[i]) / self.T_i[i]
        beta = np.sqrt((2 * np.log(self.T)) / self.T_i[i])
        self.mu_ucb[i] = min(self.mu_hat[i] + beta, 1)
        self.mu_lcb[i] = max(self.mu_hat[i] - beta, 0)

    def run(self):
        for t in range(self.T):
            if t < self.bandit.K * self.tau:  # pure exploration phase
                i = t % self.bandit.K
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.rewards[i, int(self.T_i[i] % self.tau)] = reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]
            else:  # UCB phase
                for i in range(self.bandit.K):
                    self.update_score(t, i)
                m_t = np.argmax(self.mu_lcb)
                feasible_set = [i for i in range(self.bandit.K) if self.mu_ucb[i] >= (1 - self.bandit.alpha) * self.mu_lcb[m_t]]
                i = min(feasible_set, key=lambda x: self.bandit.c[x])
                reward = np.random.binomial(1, self.bandit.mu[i])
                self.rewards[i, int(self.T_i[i] % self.tau)] = reward
                self.T_i[i] += 1
                self.costs[t] = self.bandit.c[i]

        return np.cumsum(self.costs)

