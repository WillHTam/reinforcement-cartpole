"""
Cross-Entropy
1) Simple
2) Good convergence.  Fast and works well with simple environments
    Can be used as part of a larger section.

Cross-entropy is 
    model-free: no model, just says to the agent what to do at every step 
    policy-based: approximates agent policy
    and on-policy: requires fresh data from environment

Following the ML approach and replacing all complications of the agent
    with some kind of nonlinear trainable function which maps agent's input 
    ie observations to some output. Being policy based, the nonlinear function
    (ie the NN) produces policy, and says for every observation what action the agent
    should take.

So essentially it is similar to a classification problem with the classes being equal
    to amounts of actions possible.  

Takes in episodes: ie the combination of observations, actions and their rewards.  

Due to randomness in the env and the way that the agent selections actions, 
episodes will have varying scores.

The core of cross-entropy is to throw away bad episodes and train on better ones.
1. Play N number of episodes using current model and environment.
2. Calculate total reward for every episode and decide on reward boundary
    - Usually 50th/70th percentile
3. Throw away episodes with a reward below the boundary
4. Train on remaining 'elite' episdoes using observations as the input and 
    issued actions as the desired output
5. Repeat until satisfied
"""
#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128 # 128 Hidden Neurons
BATCH_SIZE = 16 # 16 episodes played every iteration
PERCENTILE = 70 # Filter boundary, only take top 30% of episodes sorted by reward
LEARNING_RATE = 0.01

class Net(nn.Module):
    """
    One hidden layer, hyperparam tuning is not essential because of quick converge of cross-entropy

    A simple network that takes one observation from the environment as an input vector and
    outputs a number for every action performable.  Ie the probability distribution over actions.

    Instead of using softmax, use nn.CrossEntropyLoss that combines softmax and cross-entropy loss
    into a single expression.  Takes in raw non-normalized values.
        -> Requires softmax application on each take of probabilities from output.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
# two helper classes that are namedtuples from collections
#   Episode Step: Represents one single step of the agent, stores the observation 
#       from the environment and what action the agent completed. The episode steps
#       from those within the boundary will be the training data
#   Episode: Single episode stored as total undiscounted reard and a collection of
#       Episode step

def iterate_batches(env, net, batch_size):
    """
    Accepts the environment (the Env class instance from Gym), the nn, and the count
    of episodes it should generate on each iteration. 

    batch accumulates list of Episode instances (the batch)

    reward counter for the current episode

    list of steps in episode_steps

    Loop of passing observation to net, sample action to perform, and remember result of this processing
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset() # reset environment to obtain first observation
    sm = nn.Softmax(dim=1) # create softmax layer to convert network output to prob distribution
    while True:
        # At every iteration,
        obs_v = torch.FloatTensor([obs]) # convert current observation(vector of 4 numbers) to Torch tensor(shape 1x4)
                                         # by passing in as single element list
        act_probs_v = sm(net(obs_v)) # give to network to obtain action probabilities
                                     # output is raw action scores, feed through softmax 
        act_probs = act_probs_v.data.numpy()[0] 
            # unpack the gradients from softmax layer by accessing tensor.data field
            # then converting tensor to numpy array 
            # this array has same shape as input, batch dimension on axis 0
            # so take [0] to get first batch element to obtain vector of action probabilities
        action = np.random.choice(len(act_probs), p=act_probs) 
            # obtain actual action by sampling the distribution with np.random.choice()
        next_obs, reward, is_done, _ = env.step(action)
            # pass action to the environment to get next observation, reward, and ep ending indication
        episode_reward += reward # reward added to current episode's reward total
        episode_steps.append(EpisodeStep(observation=obs, action=action))
            # list of episode steps is appended with observation, action pair
            # save the observation used to choose the action NOT the observation returned by the action
        if is_done: # handles when the episode ends, in this case when the stick falls
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
                # append the finalized episode which has 
                # 1) total reward, the episode is completed and so all reward has been accumulated
                # 2) steps taken
            episode_reward = 0.0 # reset reward accumulator
            episode_steps = [] # clean list of steps
            next_obs = env.reset() # reset environment
            if len(batch) == batch_size: # if batch has reached desired number of episodes...
                yield batch # return batch to caller for processing
                    # yield transfers control to the outer loop then continues on
                batch = []
        obs = next_obs # clean up batch
            # assign observation from the environment to current observation variable

# The training of the network and generation of the episodes are perfromed at the same time.
# Every time the loop accumulates enough episodes (16), it passes control to the function caller
    # which then trains the network with gd

def filter_batch(batch, percentile):
    """
    An essential part of the cross-entropy method.
    From the given batch of episodes, calculates the boundary reward to filter
        out the best epsidoes.
    To obtain boundary, use np.percentile with list of values and the desired percentile

    Mean reward is just for monitoring
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    # filter episodes according to boundary. For every episode in the batch,
    # check that the episode has a higher total reward than the boundary
    # if it does, add to list of list of training observations and actions
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    # convert the elite observation/action lists into tensors
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    # return this tuple, last two are just for Tensorboard monitoring
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    # create environment
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # create nn, objective, optimizer, and reward boundary
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss() # the aforementioned nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward() # compute gradients
        optimizer.step() # All optimizers implement a step() method, that updates the parameters. 
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        # this last if check is the comparison of mean rewards of batch episodes
        # consider solved when over 199
        # CartPole environment will consider itself solved when mean reward for last
        # 100 episodes is over 195. But because cross entropy has quick convergence,
        # it can be stopped earlier.  
        # CartPole episode length is limited to 200.
        # If mean reward is >199, it probably means that the agent knows how to balance the stick
        if reward_m > 199: 
            print("Solved!")
            break
    writer.close()
