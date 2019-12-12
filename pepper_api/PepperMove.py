# Choregraphe bezier export in Python.
from naoqi import ALProxy
import qi
import time
import sys

ip = "192.168.0.40"
port = "9559"
STORED_VALUES = dict()
MOTOR = ["LElbowRoll", "RElbowRoll", "LElbowYaw", "RElbowYaw", "RWristYaw", "LWristYaw",
         "LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "RShoulderRoll"]
MOTOR_SIZE = len(MOTOR)
session = qi.Session()


def init():
    session.connect("tcp://" + ip + ":" + port)
    service = session.service("ALMotion")
    service.wakeUp()

    # TODO init the arms for grabbing the plate
    return service


def checkmovement(movements):
    # Check if the new movement is correct with its values.
    # Is the motor name correct?
    # Whats the difference between the old and new value?
    # Do not allow movements that are too strong
    #
    print("Checking Movements")

    for movement in movements:
        # Check if the motor exists?
        if movement[0] > MOTOR_SIZE or movement[0] < 0:
            return False
        # check if the delta to the last movement is not too high: 10 degree
        if movement[0] in STORED_VALUES:
            delta = abs(STORED_VALUES[movement[0]]
                        [0] - movements[movement[0]][0])
            if delta >= 10:
                return False
    return True


def move(movements, service):
    if (not checkmovement(movements)):
        return -1  # Bad Reward
    try:
        # The lists for sending to the robot
        names = list()  # motors list
        times = list()  # speed list
        keys = list()  # angle list
        for parm in movements:

            names.append(str(MOTOR[parm[0]]))
            keys.append(float(parm[1]))
            times.append(float(parm[2]))

            print("Moving: " + MOTOR[parm[0]] + " with: " +
                  str(parm[1]) + " in " + str(parm[2]))
            STORED_VALUES[parm[0]] = [parm[0], parm[1]]

        print((names))
        print((keys))
        print((times))

        service.angleInterpolation(names, keys, times, True)
        #service.angleInterpolation("LElbowYaw", keys, times, True)
        return 0  # No Reward at all or give some reward if the movement was correct?
    except:
        print("FAILED: %s", sys.exc_info()[0])


s = init()

args = []
# MotorID, Angle, Time
args.append([1, 1.2, 0.2])
args.append([0, -1.2, 0.2])

print(args)
move(args, s)
