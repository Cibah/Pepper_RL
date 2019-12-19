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

ip = "192.168.0.40"
port = "9559"

if __name__ == "__main__":

    pepper = init(ip, port)
    roboInit(pepper)
    winkel = dict()

    myReward = reward.Reward()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    ap.add_argument("-c", "--conf", required=True,
                    help="path to the JSON configuration file")
    args = vars(ap.parse_args())
    # filter warnings
    warnings.filterwarnings("ignore")

    event = threading.Event()  # used to stop trackingThread
    myRewardTrackerThread = rewardTrackerThread.RewardTrackerThread(
        myReward, args, event)
    # nicht mehr noetig
    # trackingThread = threading.Thread(target= myRewardTracker.run ) # , daemon= True)
    # trackingThread.start()
    myRewardTrackerThread.start()
    winkel = readAngles(pepper)
    delta = myReward.getDeltaIfNew()
    # neues Delta 1x ausgeben
    if delta is not None:
        print ("Delta= ", delta)

    #params["LShoulderPitch"] = [0.0972665, 0.96]
    params["RShoulderPitch"] = [0.0972665, 0.96]

    #params["LHand"] = [0.88, 0.96]
    #params["RHand"] = [0.88, 0.96]

    #params["LWristYaw"] = [-1.309, 0.96]
    params["RWristYaw"] = [1.409, 0.96]

    #params["LShoulderRoll"] = [0.10472, 0.96]
    params["RShoulderRoll"] = [-0.10472, 0.96]
    move(params, s)

    delta2 = myReward.getDeltaIfNew()
    winkel2 = readAngles(pepper)
    rewards = (delta - delta2)
    print("####### Run ######")
    print("Delta:\t" + str(delta))
    print("Winkel:\t" + str(winkel))
    print("Delta2:\t" + str(delta2))
    print("Winkel2:\t" + str(winkel2))
    print("rewards:\t" + str(rewards))
    # import this class # Rewards.updateRewards(delta)
