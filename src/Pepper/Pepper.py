# -*- coding: utf-8 -*-
# Choregraphe bezier export in Python.
import qi
import sys
from src.Settings import *

session = qi.Session()


def init(ip, port):
    session.connect("tcp://" + ip + ":" + port)
    service = session.service("ALMotion")
    life_service = session.service("ALAutonomousLife")
    life_service.setAutonomousAbilityEnabled("BackgroundMovement", False)
    service.wakeUp()
    return session


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
        service.angleInterpolation(names, keys, times, True)
        return 0  # No Reward at all
    except:
        print("FAILED: %s", sys.exc_info()[0])


def readAngle(session):
    service_mem = session.service("ALMemory")
    try:
        return service_mem.getData("Device/SubDeviceList/" + args['motor'] + "/Position/Sensor/Value")
    except:
        print("Error getting angle from Pepper")


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
        raw_input("Initialization completed, press enter to start:\n")
        pass
    except:
        pass
