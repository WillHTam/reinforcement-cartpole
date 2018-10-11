"""
    random movement of the cart
"""

import gym

if __name__ == '__main__':
    # create environment and initialize counter and reward accumulator
    env = gym.make('CartPole-v0')
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print('Episode done in %d steps, total reward %.2f' % (total_steps, total_reward))
    # Average of 12-15 steps before the pole falls over ie reward of 12-15
    # Most environments in Gym have a reward boundry, which is the average reward
    # that the agent should gain during 100 consecutive episodes 
    # CartPole's is 195, so it should hold the stick up for 195+ steps