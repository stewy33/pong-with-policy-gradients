"""
Super simple tests for two of the TODO's.
Passing tests doesn't indicate correctness, but it can be a helpful guide.
"""

import torch

import pong

def test_calc_discounted_future_rewards():
    rewards = [0, 0, 1, 0, 1, 1, 0, 0]
    discount = 0.99
    predicted = pong.calc_discounted_future_rewards(rewards, discount)
    true = torch.Tensor([0.9801, 0.9900, 1.0000, 0.9900, 1.0000, 1.0000, 0.0000, 0.0000])
    assert (predicted == true).all()


def test_policy_network():
    model = pong.PolicyNetwork(80 * 80, 200)

    x = torch.zeros(80 * 80)
    y = model(x)

    assert y.shape == torch.Size([1])


def main():
    test_calc_discounted_future_rewards()
    test_policy_network()


if __name__ == '__main__':
    main()
