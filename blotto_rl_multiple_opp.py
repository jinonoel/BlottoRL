import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_tools

MAX_EPISODES = 1000000
NUM_PLAYERS = 11
NUM_FIELDS = 3
NUM_COINS = 10

np.random.seed(1)

settings = {'players' : NUM_PLAYERS, 'fields' : NUM_FIELDS, 'coins' : NUM_COINS}
environment = rl_environment.Environment('blotto', **settings)
num_actions = environment.action_spec()['num_actions']
print("Possible Actions:", num_actions)
rl_agent = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions, epsilon_schedule=rl_tools.ConstantSchedule(0.2))

opponents = []
won_games = [0]
total_rewards = [0]

for i in range(1, NUM_PLAYERS):
    opponents.append(random_agent.RandomAgent(player_id=i, num_actions=num_actions))
    won_games.append(0)
    total_rewards.append(0)
    

last_probs = None

episode = 0
while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    time_step = environment.reset() #initial step per episode

    while not time_step.last():
        rl_step = rl_agent.step(time_step)
        rl_action = rl_step.action
        print('RL', environment.get_state.action_to_string(0, rl_action))
        
        actions = [rl_action]
        
        for i in range(len(opponents)):
            opp_step = opponents[i].step(time_step)
            opp_action = opp_step.action
            actions.append(opp_action)
            
            print('Opp', environment.get_state.action_to_string(1, opp_action))
            
        last_probs = rl_step
        
        time_step = environment.step(actions)

    #Episode over, step all agents with final state
    rl_agent.step(time_step)

    for i in range(len(opponents)):
        opponents[i].step(time_step)

    rewards = environment.get_state.returns()
    print ("Rewards:", rewards)

    for i in range(len(rewards)):
        #count only positive rewards as won games
        won_games[i] += rewards[i] if rewards[i] > 0 else 0
        total_rewards[i] += rewards[i] 

    print()
    
print("\nWON Games")
print("RL Agent:", int(won_games[0]), total_rewards[0])
for i in range(1, NUM_PLAYERS):
    print('Opponent:', int(won_games[i]), total_rewards[i])


q_list = []
for act in rl_agent._q_values['[0.0]'].keys():
    q_val = rl_agent._q_values['[0.0]'][act]
    q_list.append(q_val)

done = set()
for val in sorted(q_list, reverse=True):
    if val in done:
        continue

    done.add(val)

    for act in rl_agent._q_values['[0.0]'].keys():
        if val == rl_agent._q_values['[0.0]'][act]:
            act_string = environment.get_state.action_to_string(0, act)
            print(val, act, act_string)
