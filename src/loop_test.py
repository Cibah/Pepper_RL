import json

with open('Pepper_Training.txt') as json_file:
    data = json.load(json_file)
    q = data['steps']
    for x in range(len(q)):
        p = q[x]
        s = [p['az'], p['ad']]
        print(s)
        # Kombiniere az mit ad ?
        # s = az + ad

        # s2 = fz + fd


        s2 = [p['fz'], p['fd']]
        print(s2)
        r = p['rw']
        # info = False Wird nicht benutzt??

