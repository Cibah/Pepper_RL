# -*- coding: utf-8 -*-
import json
from src.Settings import *


def saveData(data):
    f = open(TRAINING_FILE, "a")
    f.write(data)
    f.close()


def readData():
    f = open(TRAINING_FILE, "r")
    print(f.read())


def load():
    with open(TRAINING_FILE) as json_file:
        data = json.load(json_file)
        q = data['steps']

        for x in range(len(q)):
            p = q[x]
            print('Action: ' + str(p['action']))
            print('Anfangszustand: ' + str(p['az']))
            print('Anfangsdelta: ' + str(p['ad']))
            print('Folgezustand: ' + str(p['fz']))
            print('Folgedelta: ' + str(p['fd']))
            print('Reward: ' + str(p['rw']))


class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
