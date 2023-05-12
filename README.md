# Multi-Armed Bandits with Cost Subsidy

## Description

This project provides an implementation of three algorithms: Upper Confidence Bound (UCB), Thompson Sampling (TS), and Explore-Then-Commit (ETC) for the Multi-Armed Bandit (MAB) problem with cost subsidy. The goal of the project is to replicate and understand the results presented in the paper "Multi-armed Bandits with Cost Subsidy" (Sinha et al., 2020).

WARNING: This is still under development!

## Structure

The project is organized into four Python files:

`bandit.py`: Contains the Bandit class that represents the multi-armed bandit problem.

`ucb.py`: Contains the UCB class that implements the Upper Confidence Bound algorithm.

`ts.py`: Contains the TS class that implements the Thompson Sampling algorithm.

`etc.py`: Contains the ETC class that implements the Explore-Then-Commit algorithm.

Each of these files defines a class that can be used in a main script to simulate the MAB problem with cost subsidy.

## How to Run

To run a simulation, you need to run `python main.py`.

## Pseudo Algorithms

Here are the pseudo algorithms from the original paper:

<img src="https://github.com/zoeshao0425/MAB-with-cost-subsidy/blob/main/assets/pseudo_alg1.jpg" width="400">
<img src="https://github.com/zoeshao0425/MAB-with-cost-subsidy/blob/main/assets/pseudo_alg2.jpg" width="400">


## Results in the Original Paper

Here are the results from the original paper:

<img src="https://github.com/zoeshao0425/MAB-with-cost-subsidy/blob/main/assets/figure1.jpg" width="500">

## Reference
Sinha, D., Sankararaman, K. A., Kazerouni, A., & Avadhanula, V. (2020). Multi-armed Bandits with Cost Subsidy. arXiv preprint arXiv:2011.01488. Retrieved from https://arxiv.org/abs/2011.01488
