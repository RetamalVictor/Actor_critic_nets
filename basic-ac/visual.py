import gym
import numpy as np
from actor_critic import Agent


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(alpha=0.003, gamma=0.99, n_actions=env.action_space.n)
    n_games = 50
    scores = []
    best_score = env.reward_range[0]

    load_checkpoint = True

    if load_checkpoint:
        agent.load_checkpoint()

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            env.render()
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, next_observation, done)
            observation = next_observation
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print(f"episode: {i} score: {score} avg_score: {avg_score}")
    print("Best score: ", best_score)
