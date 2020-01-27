# -*- coding:utf-8 -*-
# testBallTracker.py
# kleine Test-Applikation, um die reward-Funktionalitaet des BallTrackers zu testen
# nutzt aber jetzt RewardTrackerThread-Klasse, d.h. abgeleitete Thread-Klasse
#
import argparse
import warnings
import time
import threading
import src.BallTracker.reward
import src.BallTracker.rewardTrackerThread
from src.Settings import *


class BallTrackerThread(threading.Thread):
    delta = "START"
    exitFlag = 0

    def __init__(self):
        threading.Thread.__init__(self)

    def updateDelta(self, input):
        self.delta = input

    def run(self):
        myReward = src.BallTracker.reward.Reward()
        # myReward.print()
        # signal.signal(signal.SIGINT, shutdown)

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video",
                        help="path to the (optional) video file", default=VIDEODEVICE)
        ap.add_argument("-b", "--buffer", type=int, default=64,
                        help="max buffer size")
        ap.add_argument("-c", "--conf",
                        help="path to the JSON configuration file", default=BALLTRACKERCONFIG)
        args = vars(ap.parse_args())
        # filter warnings
        warnings.filterwarnings("ignore")

        event = threading.Event()  # used to stop trackingThread
        myRewardTrackerThread = src.BallTracker.rewardTrackerThread.RewardTrackerThread(myReward, args, event)
        myRewardTrackerThread.start()

        while self.exitFlag == 0:
            deltaTMP = myReward.getDeltaIfNew()
            if deltaTMP is not None:
                self.delta = deltaTMP
            time.sleep(0.01)
