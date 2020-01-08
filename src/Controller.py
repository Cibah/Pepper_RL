import argparse
import warnings
import time
import threading
import sys			# used for exit()
import signal			# to catch Ctrl-C Interrupt
import readAngles
import pepperMove
import socket
import thread
import json

ip = "192.168.0.40"
port = "9559"

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))


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

def getDelta():
    global delta
    while True:
        data, addr = sock.recvfrom(128) # buffer size is 1024 bytes
        #print "received message:", data
        delta= data.decode()
        #print(delta)

def getReward(input1, input2):

    input1 = input1.replace("(","")
    input2 = input2.replace("(","")
    
    input1 = input1.replace(")","")
    input2 = input2.replace(")","")

    var1_x = input1.partition(",")[0]
    var1_y = input1.partition(",")[2]
    var2_x = input2.partition(",")[0]
    var2_y = input2.partition(",")[2]
    print("DEBUG: " + var1_x)    
    #print("TYPE: " + type(var1_x))
    var1_x = (abs(int(var1_x)))
    var2_x = (abs(int(var2_x)))
    var1_y = (abs(int(var1_y)))
    var2_y = (abs(int(var2_y)))

    sum_x = var1_x + var2_x 
    sum_y = var1_y + var2_y 
    reward = 100 - (sum_x + sum_y)
    return reward

if __name__ == "__main__":

    global delta

    t1 = threading.Thread(target=getDelta)
    t1.start()
    print("udp thread running...")
    session = pepperMove.init(ip, port)
    pepperMove.roboInit(session)
    winkel = dict()

    winkel = readAngles.readAngles(session)
    delta1 = delta
    #Winkel aus choreographe whlen
    params = dict()
    params["RShoulderPitch"] = [0.0872665, 0.96]
    params["LShoulderPitch"] = [0.460767, 0.96]
    params["LHand"] = [0.88, 0.96]
    params["RHand"] = [0.88, 0.96]

    params["LWristYaw"] = [-1.309, 0.96]
    params["RWristYaw"] = [1.309, 0.96]

    params["LShoulderRoll"] = [0.10472, 0.96]
    params["RShoulderRoll"] = [-0.10472, 0.96]

    #params["RShoulderPitch"] = [0.1272665, 0.3]

    #params["LHand"] = [0.88, 0.96]
    #params["RHand"] = [0.88, 0.96]

    #params["LWristYaw"] = [-1.309, 0.96]
    #params["RWristYaw"] = [2.409, 0.3]

    #params["LShoulderRoll"] = [0.10472, 0.96]
    #params["RShoulderRoll"] = [0.30472, 0.3]
    
    service = session.service("ALMotion")
    pepperMove.move(params, service)

    delta2 = delta
    winkel2 = readAngles.readAngles(session)
    print("rewards= "+ delta1+" - "+ delta2)
    #rewards = int((int(delta) - int(delta2)))
    rewards = delta
    print("####### Run ######")
    print("Delta:\t" + str(delta))
    print("Winkel:\t" + str(winkel))
    print("Delta2:\t" + str(delta2))
    print("Winkel2:\t" + str(winkel2))
    print("rewards:\t" + str(rewards))
    
    me = Object()
    me.az = winkel
    me.ad = delta1
    me.fz = winkel2
    me.fd = delta2
    me.rw = getReward(delta1,delta2)
    exportData = me.toJSON()
    saveData(exportData)

exit()
