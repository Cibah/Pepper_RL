import json

with open('training_data.txt') as json_file:
    data = json.load(json_file)
    q = data['steps']
    for x in range(len(q)):
        p = q[x]
        s = [p['az'].substring(":"), p['ad'].substring(0, ',')]
        print(s)
        # Kombiniere az mit ad ?
        # s = az + ad

        # s2 = fz + fd


        s2 = [p['fz'].substring(":"), p['fd'].substring(0, ',')]
        print(s2)
        r = p['rw']
        # info = False Wird nicht benutzt??

