# -*- coding: utf-8 -*-
# Choregraphe bezier export in Python.
import qi
import sys

STORED_VALUES = dict()
MOTORS = ["LElbowRoll", "RElbowRoll", "LElbowYaw", "LWristYaw", "RWristYaw",
          "RElbowYaw", "LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "RShoulderRoll", "LHand", "RHand"]
session = qi.Session()


def init(ip, port):
    session.connect("tcp://" + ip + ":" + port)
    service = session.service("ALMotion")
    # service.wakeUp()
    # service.rest()
    life_service = session.service("ALAutonomousLife")
    life_service.setAutonomousAbilityEnabled("BackgroundMovement", False)
    service.wakeUp()
    return session


def checkMovement(movements):
    # Check if the new movement is correct with its values.
    # Is the motor name correct?
    # Whats the difference between the old and new value?
    # Do not allow movements that are too strong
    #
    #print("Checking Movements")

    for motor in movements:
        #print(motor)
        if motor not in MOTORS:
            print("Error in check: Motor <" + motor + "> not found")
            # return False
        if motor in STORED_VALUES:
            delta = abs(STORED_VALUES[motor][0] - movements[motor][0])
            if delta >= 10:
                print("Error in check: delta too high: " + delta)
                return False
    return True


def move(movements, service):
    try:
        names = list()
        times = list()
        keys = list()
        for parm in movements:
            names.append(str(parm))
            keys.append(float(movements[parm][0]))
            # time is thee duration of the movement
            times.append(float(movements[parm][1]))

            STORED_VALUES[parm] = [movements[parm][0], movements[parm][1]]
        service.angleInterpolation(names, keys, times, True)
        return 0  # No Reward at all
    except:
        print("FAILED: %s", sys.exc_info()[0])


def readAngles(session):
    MOTORS = ["RShoulderPitch"]
    service_mem = session.service("ALMemory")
    angles = dict()
    for motor in MOTORS:
        link = "Device/SubDeviceList/" + motor + "/Position/Sensor/Value"
        exp = service_mem.getData(link)
        angles[motor] = exp
    return angles


def roboInit(session):
    service = session.service("ALMotion")

    params = dict()

    params["LShoulderPitch"] = [0.088, 0.96]
    params["RShoulderPitch"] = [0.088, 0.96]

    params["LHand"] = [0.88, 0.96]
    params["RHand"] = [0.88, 0.96]

    params["LWristYaw"] = [-1.309, 0.96]
    params["RWristYaw"] = [1.309, 0.96]

    params["LShoulderRoll"] = [0.10472, 0.96]
    params["RShoulderRoll"] = [-0.10472, 0.96]
    # params["HipPitch"] = [-0.185005, 0.96]
    move(params, service)
    try:
        raw_input("Initialization completed, press enter:\n")
    except:
        pass
