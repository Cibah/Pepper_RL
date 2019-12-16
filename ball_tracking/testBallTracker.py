# testBallTracker.py
# kleine Test-Applikation, um die reward-Funktionalität des BallTrackers zu testen
# nutzt aber jetzt RewardTrackerThread-Klasse, d.h. abgeleitete Thread-Klasse
#
import argparse
import warnings
import time
import threading
import reward
import rewardTrackerThread
import sys			# used for exit()
import signal			# to catch Ctrl-C Interrupt
#import ddpg

# define little signal Handler to shutdown rewardTracker-Thread, when Main-Thread closes
def shutdown(sig, frame):
    print ("Closing...")
    event.set()
    # trackingThread.join()	# not requires because of event
    sys.exit(0)

if __name__ == "__main__":
    myReward = reward.Reward()
    # myReward.print()
    signal.signal(signal.SIGINT, shutdown)

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
    
    event = threading.Event()	# used to stop trackingThread 
    myRewardTrackerThread = rewardTrackerThread.RewardTrackerThread( myReward, args, event)
    # nicht mehr nötig
    # trackingThread = threading.Thread(target= myRewardTracker.run ) # , daemon= True)
    # trackingThread.start()
    myRewardTrackerThread.start() 

    while True :
        delta= myReward.getDeltaIfNew();
        # neues Delta 1x ausgeben
        if delta is not None :
           print ("Reward= ", delta)
           #import this class # Rewards.updateRewards(delta)
        time.sleep(0.01)

    shutdown(signal.SIGINT, None)
# Ende
