import argparse
import warnings
import time
import threading
import reward
import rewardTrackerThread
import sys			# used for exit()
import signal			# to catch Ctrl-C Interrupt
import readAngles
import pepperMove
import ballTracker


ip = "192.168.0.41"
port = "9559"

# define little signal Handler to shutdown rewardTracker-Thread, when Main-Thread closes


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    ap.add_argument("-c", "--conf", required=True,
                    help="path to the JSON configuration file")
    args = vars(ap.parse_args())
    event = threading.Event()
    # Als Thread laufen lassen
    myBallTrackerThread = ballTracker.BallTracker(args, event)
    ballTracker.ballRunner()

    session = pepperMove.init(ip, port)
    pepperMove.roboInit(session)
    winkel = dict()

    myReward = reward.Reward()

    winkel = readAngles.readAngles(session)
    delta = ballTracker.delta
    # neues Delta 1x ausgeben
    params = dict()
    # params["LShoulderPitch"] = [0.0972665, 0.96]
    params["RShoulderPitch"] = [0.0972665, 0.96]

    # params["LHand"] = [0.88, 0.96]
    # params["RHand"] = [0.88, 0.96]

    # params["LWristYaw"] = [-1.309, 0.96]
    params["RWristYaw"] = [1.409, 0.96]

    # params["LShoulderRoll"] = [0.10472, 0.96]
    params["RShoulderRoll"] = [-0.10472, 0.96]
    service = session.service("ALMotion")
    pepperMove.move(params, service)

    delta2 = ballTracker.delta
    winkel2 = readAngles.readAngles(session)
    rewards = int((int(delta) - int(delta2)))
    print("####### Run ######")
    print("Delta:\t" + str(delta))
    print("Winkel:\t" + str(winkel))
    print("Delta2:\t" + str(delta2))
    print("Winkel2:\t" + str(winkel2))
    print("rewards:\t" + str(rewards))
    # import this class # Rewards.updateRewards(delta)
    shutdown(signal.SIGINT, None)


def funktion(session):
