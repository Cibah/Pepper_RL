# Choregraphe bezier export in Python.
#from naoqi import ALProxy
import qi
import time
import sys

def readAngles():
    ip = "192.168.0.40"
    port = "9559"
    STORED_VALUES = dict()
    MOTORS = ["LElbowRoll", "RElbowRoll", "LElbowYaw","LWristYaw", "RWristYaw",
              "RElbowYaw", "LShoulderPitch", "RShoulderPitch", "LShoulderRoll", "RShoulderRoll", "LHand", "RHand"]
    session = qi.Session()
    session.connect("tcp://" + ip + ":" + port)
    service_mem= session.service("ALMemory")
    angles = dict()
    for motor in MOTORS:
	link="Device/SubDeviceList/"+motor+"/Position/Sensor/Value"
    	exp = service.getData(link)
	angles[motor] = exp    	
	#print(link)
    	#print("GRAD : ", exp*180/3.141)
    	#print("RAD: ", exp)    

    return angles


