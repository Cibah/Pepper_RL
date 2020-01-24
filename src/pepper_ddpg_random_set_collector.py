# -*- coding: utf-8 -*-
## Reines Sammeln von Trainingsdaten mit Hilfe von Zufallszahlen

from src.Pepper import Pepper
from src.Pepper.Pepper import readAngle
from src.Settings import *
from src.ddpg.ddpg import getReward
from src.BallTracker import ballTracker
import random

from src.files.files import Object, saveData

if __name__ == "__main__":
    print("Generating random movement sets")
    global delta

    thread1 = ballTracker.BallTrackerThread()
    thread1.start()

    session = Pepper.init(ip, port)
    Pepper.roboInit(session)
    params = dict()

    for x in range(TRAINING_STEPS):
        # Zufallswert fuer eine Bewegung.
        winkelToTrain1 = float("%.2f" % round(random.uniform(-1.0, 1.0), 2))

        # Bewegung innerhalb der Grenzwerte? Wenn nicht: Strafe
        threshold_reward = 0
        if winkelToTrain1 <= UNTERE_GRENZE:
            winkelToTrain1 = UNTERE_GRENZE
            threshold_reward = -10000

        if winkelToTrain1 >= OBERE_GRENZE:
            winkelToTrain1 = OBERE_GRENZE
            threshold_reward = -10000

        params[args['motor']] = [winkelToTrain1, TIME_TO_MOVE]
        service = session.service("ALMotion")

        # Holfe Anfangszustand
        delta1 = thread1.delta[0]
        winkel = readAngle(session)

        # Bewege Arm
        Pepper.move(params, service)

        # Hole Folgezustand
        delta = thread1.delta[0]
        delta2 = delta
        winkel2 = readAngle(session)

        me = Object()
        me.steps.az = winkel
        me.steps.ad = delta1
        me.steps.actionR = winkelToTrain1
        me.steps.fz = winkel2
        me.steps.fd = delta2
        me.steps.rw = getReward(delta2) + threshold_reward

        exportData = me.toJSON()
        saveData(exportData)
    print("Fertig")
    # thread1.exitFlag
    # TODO Beende Thread
    # thread1.join()
exit()
