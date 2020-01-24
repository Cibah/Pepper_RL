# testBallTracker.py
# kleine Test-Applikation, um die reward-Funktionalität des BallTrackers zu testen
# nutzt aber jetzt RewardTrackerThread-Klasse, d.h. abgeleitete Thread-Klasse
#
import warnings
import time
import threading
import src.BallTracker.reward
import src.BallTracker.rewardTrackerThread
import sys  # used for exit()
import signal  # to catch Ctrl-C Interrupt


# define little signal Handler to shutdown rewardTracker-Thread, when Main-Thread closes


class BallTracker(threading.Thread):

    def __init__(self, arguments, event):
        threading.Thread.__init__(self)
        self.args = arguments
        self.delta = (0.0, 0.0)

    def shutdown(sig, frame):
        print ("Closing...")
        event.set()
        # trackingThread.join()	# not requires because of event
        sys.exit(0)

    def ballRunner():
        myReward = src.BallTracker.reward.Reward()
        # myReward.print()
        signal.signal(signal.SIGINT, shutdown)
        # filter warnings
        warnings.filterwarnings("ignore")

        event = threading.Event()  # used to stop trackingThread
        myRewardTrackerThread = src.BallTracker.rewardTrackerThread.RewardTrackerThread(
            myReward, args, event)
        # nicht mehr nötig
        # trackingThread = threading.Thread(target= myRewardTracker.run ) # , daemon= True)
        # trackingThread.start()
        myRewardTrackerThread.start()

        while True:
            self.delta = myReward.getDeltaIfNew()
            # neues Delta 1x ausgeben
            if self.delta is not None:
                print ("Delta= ", self.delta)
                # import this class # Rewards.updateRewards(delta)
            time.sleep(0.01)

        shutdown(signal.SIGINT, None)
# Ende
