import readAngles
import pepperMove
import json
import ballTracker
import random

ip = "192.168.0.40"
port = "9559"

delta = ""


class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def saveData(data):
    f = open("training_data.txt", "a")
    f.write(data)
    f.close()


def readData():
    f = open("training_data.txt", "r")
    print(f.read())


def getReward(delta):
    #print("Delta: " + str(delta))
    delta = str(delta).replace("(", "")
    delta = delta.replace(")", "")
    var2_x = delta.partition(",")[0]
    var2_y = delta.partition(",")[2]
    # print("TYPE: " + type(var1_x))
    var2_x = (abs(int(var2_x)))
    var2_y = (abs(int(var2_y)))
    reward = 100 - (var2_x + var2_y)
    return reward


if __name__ == "__main__":
    print("Starte BallTrackerThread")
    global delta

    thread1 = ballTracker.BallTrackerThread()
    thread1.start()

    print("Main running...")
    session = pepperMove.init(ip, port)
    pepperMove.roboInit(session)
    winkel = dict()

    winkel = readAngles.readAngles(session)
    # Winkel aus choreographe whlen
    params = dict()

    TRAINING_STEPS = 50000
    OBERE_GRENZE = 0.46
    UNTERE_GRENZE = -0.095
    TIME_TO_MOVE = 0.3

    for x in range(TRAINING_STEPS):
        winkelToTrain1 = random.uniform(UNTERE_GRENZE, OBERE_GRENZE)

        params["RShoulderPitch"] = [winkelToTrain1, TIME_TO_MOVE]

        service = session.service("ALMotion")
        print("Bewege Motor RShoulderPitch um " + str(winkelToTrain1))

        delta1 = thread1.delta
        pepperMove.move(params, service)
        delta = thread1.delta
        delta2 = delta
        winkel2 = readAngles.readAngles(session)
        #print("rewards= " + str(delta1) + " - " + str(delta2))
        # rewards = int((int(delta) - int(delta2)))
        rewards = delta

        me = Object()
        me.az = winkel
        me.ad = delta1
        me.actionR = winkelToTrain1
        me.fz = winkel2
        me.fd = delta2
        me.v = float(float(delta2[0]) - float(delta[0]))
        print(str(me.v))
        me.rw = getReward(delta2)

        exportData = me.toJSON()
        saveData(exportData)
    #thread1.delta.exitFlag = 1
    print("Fertig")
    #thread1.join()
exit()
