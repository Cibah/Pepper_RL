import json

with open('training_data.txt') as json_file:
    data = json.load(json_file)
    q=data['steps'] 
        
    for x in range(len(q)):
        p=q[x]
        print('Action: ' + str(p['action']))
        print('Anfangszustand: ' + str(p['az']))
        print('Anfangsdelta: ' + str(p['ad']))
        print('Folgezustand: ' + str(p['fz']))
        print('Folgedelta: ' + str(p['fd']))
        print('Reward: ' + str(p['rw']))
