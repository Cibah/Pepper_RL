# Choregraphe bezier export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names.append("HeadPitch")
times.append([0.32])
keys.append([[0.0238194, [3, -0.12, 0], [3, 0, 0]]])

names.append("HeadYaw")
times.append([0.32])
keys.append([[0.00274, [3, -0.12, 0], [3, 0, 0]]])

names.append("HipPitch")
times.append([0.32])
keys.append([[-0.0426928, [3, -0.12, 0], [3, 0, 0]]])

names.append("HipRoll")
times.append([0.32])
keys.append([[-0.00642528, [3, -0.12, 0], [3, 0, 0]]])

names.append("KneePitch")
times.append([0.32])
keys.append([[-0.00263859, [3, -0.12, 0], [3, 0, 0]]])

names.append("LElbowRoll")
times.append([0.32])
keys.append([[-0.500858, [3, -0.12, 0], [3, 0, 0]]])

names.append("LElbowYaw")
times.append([0.32])
keys.append([[-1.20887, [3, -0.12, 0], [3, 0, 0]]])

names.append("LHand")
times.append([0.32])
keys.append([[0.39475, [3, -0.12, 0], [3, 0, 0]]])

names.append("LShoulderPitch")
times.append([0.32])
keys.append([[1.57598, [3, -0.12, 0], [3, 0, 0]]])

names.append("LShoulderRoll")
times.append([0.32])
keys.append([[0.142349, [3, -0.12, 0], [3, 0, 0]]])

names.append("LWristYaw")
times.append([0.32])
keys.append([[-0.112267, [3, -0.12, 0], [3, 0, 0]]])

names.append("RElbowRoll")
times.append([0.32])
keys.append([[0.512191, [3, -0.12, 0], [3, 0, 0]]])

names.append("RElbowYaw")
times.append([0.32])
keys.append([[1.21035, [3, -0.12, 0], [3, 0, 0]]])

names.append("RHand")
times.append([0.32])
keys.append([[0.386515, [3, -0.12, 0], [3, 0, 0]]])

names.append("RShoulderPitch")
times.append([0.32])
keys.append([[1.57448, [3, -0.12, 0], [3, 0, 0]]])

names.append("RShoulderRoll")
times.append([0.32])
keys.append([[-0.150911, [3, -0.12, 0], [3, 0, 0]]])

names.append("RWristYaw")
times.append([0.32])
keys.append([[0.116703, [3, -0.12, 0], [3, 0, 0]]])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  # motion = ALProxy("ALMotion", IP, 9559)
  motion = ALProxy("ALMotion")
  motion.angleInterpolationBezier(names, times, keys)
except BaseException, err:
  print err

