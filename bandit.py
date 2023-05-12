import numpy as np

class Bandit:
    def __init__(self, K, mu, c, alpha):
        self.K = K
        self.mu = mu
        self.c = c
        self.alpha = alpha
        self.m_star = np.argmax(self.mu)