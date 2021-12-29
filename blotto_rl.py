import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

MAX_EPISODES = 1000
NUM_PLAYERS = 2
NUM_FIELDS = 3
NUM_COINS = 10

np.random.seed(1)

settings = {'players' : NUM_PLAYERS, 'fields' : NUM_FIELDS, 'coins' : NUM_COINS}
environment = rl_environment.Environment('blotto', **settings)
num_actions = environment.action_spec()['num_actions']
print("Possible Actions:", num_actions)
rl_agent = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions)
opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

won_games = [0,0]

episode = 0
while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    time_step = environment.reset() #initial step per episode

    while not time_step.last():
        rl_step = rl_agent.step(time_step)
        opp_step = opponent.step(time_step)

        rl_action = rl_step.action
        opp_action = opp_step.action

        print('RL', rl_action, ':', environment.get_state.action_to_string(0, rl_action))
        print('Opp', opp_action, ':', environment.get_state.action_to_string(1, opp_action))
        actions = [rl_action, opp_action]

        time_step = environment.step(actions)

    #Episode over, step both agents with final state
    rl_agent.step(time_step)
    opponent.step(time_step)

    #print(environment.get_state)
    rewards = environment.get_state.returns()
    print ("Rewards:", rewards)
    
    for player in range(NUM_PLAYERS):
        won_games[player] += rewards[player] if rewards[player] > 0 else 0

    print()
    
print("\nWON Games")
print("RL Agent:", int(won_games[0]))
print('Opponent:', int(won_games[1]))
