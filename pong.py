"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
Modified from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

"""

import argparse
import datetime as datetime
import math
import os
import random
import time

import gym
import tensorflow as tf
import torch
import torch.nn as nn

def preprocess(image):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """

    image = torch.Tensor(image)

    # Crop, downsample by factor of 2, and turn to grayscale by keeping only red channel
    image = image[35:195]
    image = image[::2,::2, 0]

    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1

    return image.flatten().float()


def calc_discounted_future_rewards(rewards, discount_factor):
    """
    Calculate the discounted future reward at each timestep.

    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]

    """

    discounted_future_rewards = torch.empty(len(rewards))

    # Compute discounted_future_reward for each timestep by iterating backwards
    # from end of episode to beginning
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        if rewards[t] != 0:
            discounted_future_reward = 0

        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards


class PolicyNetwork(nn.Module):
    """ Simple two-layer MLP for policy network. """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        prob_up = torch.sigmoid(x)

        return prob_up


def run_episode(model, env, discount_factor, render=False):
    UP = 2
    DOWN = 3

    observation = env.reset()
    prev_x = preprocess(observation)

    action_chosen_log_probs = []
    rewards = []

    done = False
    timestep = 0

    while not done:
        if render:
            # Render game window at 30fps
            time.sleep(1 / 30)
            env.render()

        # Preprocess the observation, set input to network to be difference
        # image between frames
        cur_x = preprocess(observation)
        x = cur_x - prev_x
        prev_x = cur_x

        # Run the policy network and sample action from the returned probability
        prob_up = model(x)
        action = UP if random.random() < prob_up else DOWN # roll the dice!

        # Calculate the probability of sampling the action that was chosen
        action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        action_chosen_log_probs.append(torch.log(action_chosen_prob))

        # Step the environment, get new measurements, and updated discounted_reward
        observation, reward, done, info = env.step(action)
        rewards.append(torch.Tensor([reward]))
        timestep += 1

    # Concat lists of log probs and rewards into 1-D tensors
    action_chosen_log_probs = torch.cat(action_chosen_log_probs)
    rewards = torch.cat(rewards)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It encourages roughly half of the actions to be rewarded and half to be discouraged, which
    # is helpful especially in beginning when positive reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) \
                                     / discounted_future_rewards.std()

    # PG magic happens right here, multiplying action_chosen_log_prob by future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()

    return loss, rewards.sum()


def train(render=False):
    # Hyperparameters
    input_size = 80 * 80 # input dimensionality: 80x80 grid
    hidden_size = 200 # number of hidden layer neurons
    learning_rate = 1e-4
    discount_factor = 0.99 # discount factor for reward

    batch_size = 4
    save_every_batches = 5

    # Load model from checkpoint if exists, otherwise, initialize new model
    if os.path.exists('policy_network.pth'):
        save_dict = torch.load('policy_network.pth')

        model = save_dict['model']
        optimizer = save_dict['optimizer']
        start_time = save_dict['start_time']
        last_batch = save_dict['last_batch']
    else:
        model = PolicyNetwork(input_size, hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        last_batch = -1

    # Set up tensorboard logging
    tf_writer = tf.summary.create_file_writer(f'tensorboard_logs/{start_time}')
    tf_writer.set_as_default()

    # Create pong environment (PongDeterministic versions run faster)
    env = gym.make("PongDeterministic-v4")

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    while True:

        mean_batch_loss = 0
        mean_batch_reward = 0
        for batch_episode in range(batch_size):

            # Run one episode
            loss, episode_reward = run_episode(model, env, discount_factor, render)
            mean_batch_loss += loss / batch_size
            mean_batch_reward += episode_reward / batch_size

            # Boring book-keeping
            print(f'Episode reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        optimizer.zero_grad()
        mean_batch_loss.backward()
        optimizer.step()

        # Batch metrics and tensorboard logging
        print(f'Batch: {batch}, mean loss: {mean_batch_loss:.2f}, mean reward: {mean_batch_reward:.2f}')
        tf.summary.scalar('mean loss', mean_batch_loss.detach().item(), step=batch)
        tf.summary.scalar('mean reward', mean_batch_reward.detach().item(), step=batch)

        if batch % save_every_batches == 0:
            print('Saving checkpoint...')
            save_dict = {
                'model': model,
                'optimizer': optimizer,
                'start_time': start_time,
                'last_batch': batch
            }
            torch.save(save_dict, 'policy_network.pth')

        batch += 1


def main():
    # By default, doesn't render game screen, but can invoke with `--render` flag on CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    train(render=args.render)


if __name__ == '__main__':
    main()