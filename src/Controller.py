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

if __name__ == "__main__":
    myReward = reward.Reward()
