import pyspiel
import numpy as np

GAME = pyspiel.load_game('blotto', {'players' : 2, 'fields' : 3})
MAX_EPISODES = 100


numPlayers = GAME.num_players()

scores = []
for n in range(numPlayers):
    scores.append(0)



episode = 0

while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    state = GAME.new_initial_state()

    while not state.is_terminal():
        #    sddsffsd

        actions = []
        
        for player in range(numPlayers):
            action = np.random.choice(state.legal_actions(player))
            #print(state.action_to_string(player, action))
            
            actions.append(action)

        state.apply_actions(actions)

        print(str(state))

    returns = state.returns()
    print ("Returns:", returns)
    
    for player in range(numPlayers):
        scores[player] += returns[player]

    print()
    
print("\nFINAL SCORES")
for player in range(numPlayers):
    print("Player", player, ":", scores[player])
