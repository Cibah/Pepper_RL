# Reines Sammeln von Trainingsdaten mit Hilfe von Zufallszahlen

from src.Pepper import Pepper
from src.Pepper.Pepper import readAngles
from src.Settings import *
from src.ddpg.ddpg import getReward
from src.BallTracker import ballTracker
import random

from src.files.files import Object, saveData

if __name__ == "__main__":
    print("Starte BallTrackerThread")
    global delta

    thread1 = ballTracker.BallTrackerThread()
    thread1.start()

    print("Main running...")
    session = Pepper.init(ip, port)
    Pepper.roboInit(session)

    params = dict()

    for x in range(TRAINING_STEPS):
        winkelToTrain1 = float("%.2f" % round(random.uniform(-1.0, 1.0), 2))

        rewardTMP = 0
        if winkelToTrain1 <= UNTERE_GRENZE:
            winkelToTrain1 = UNTERE_GRENZE
            rewardTMP = -10000

        if winkelToTrain1 >= OBERE_GRENZE:
            winkelToTrain1 = OBERE_GRENZE
            rewardTMP = -10000

        params[args['motor']] = [winkelToTrain1, TIME_TO_MOVE]
        service = session.service("ALMotion")
        print("Bewege Motor " + args['motor'] + " um " + str(winkelToTrain1))
        delta1 = thread1.delta[0]
        winkel = readAngles(session).get(args['motor'])

        Pepper.move(params, service)

        delta = thread1.delta[0]
        delta2 = delta
        winkel2 = readAngles(session).get(args['motor'])
        # print("rewards= " + str(delta1) + " - " + str(delta2))
        # rewards = int((int(delta) - int(delta2)))
        rewards = delta

        me = Object()
        me.az = winkel
        me.ad = delta1
        me.actionR = winkelToTrain1
        me.fz = winkel2
        me.fd = delta2
        me.v = 0.0
        # float(float(delta2[0]) - float(delta[0]))/TIME_TO_MOVE
        # print(str(me.v))
        me.rw = getReward(delta2) + rewardTMP

        exportData = me.toJSON()
        saveData(exportData)
    print("Fertig")
    #thread1.join()
exit()
