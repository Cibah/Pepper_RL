# Choregraphe bezier export in Python.
# from naoqi import ALProxy
import qi
import time
import sys


def readAngles(session):
    STORED_VALUES = dict()
    MOTORS = ["RShoulderPitch"]
    # session = qi.Session()
    # session.connect("tcp://" + ip + ":" + port)
    service_mem = session.service("ALMemory")
    angles = dict()
    for motor in MOTORS:
        link = "Device/SubDeviceList/" + motor + "/Position/Sensor/Value"
        exp = service_mem.getData(link)
        angles[motor] = exp
        # print(link)
        # print("GRAD : ", exp*180/3.141)
        # print("RAD: ", exp)

    return angles
