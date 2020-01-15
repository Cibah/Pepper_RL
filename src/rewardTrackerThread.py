# rewardTrackerThread.py Klasse, die z.B. mit einer Kamera die aktuelle
# Abweichung von der Zielvorgabe erfasst und als "Reward" zur Verfuegung stellt.
# Sie nutzt dazu eine Instanz der Reward-Klasse, in die das aktuelle Ergebnis
# geschrieben wird.
# Bekommt args beim Anlegen, um auf command-line Argumente zuzugreifen
# weiterentwickelt von rewardTracker.py: jetzt als abgeleitete Thread-Klasse

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import commentjson
import cv2
import imutils
import time
import reward
import threading  # Parent Class
import signal
import os  # required to get PID to send signal to Main Thread, if Tracker closes


class RewardTrackerThread (threading.Thread):
    def __init__(self, reward, arguments, event):
        threading.Thread.__init__(self)  # call init of Parent-Class "Thread"
        self.event = event
        self.reward = reward
        self.arguments = arguments

        self.conf = commentjson.load(open("./conf.json"))
        # init Farbwerte von grosser Kugel und kleinem Ball
        # Achtung: Upper / Lower in HSV.Format!!
        self.color1 = eval(self.conf["color1"])
        self.lower1 = eval(self.conf["lower1"])
        self.upper1 = eval(self.conf["upper1"])
        self.color2 = eval(self.conf["color2"])
        self.lower2 = eval(self.conf["lower2"])
        self.upper2 = eval(self.conf["upper2"])
        # Punkte fuer "Leucht-Schwanz" des kleinen Balls
        self.pts = deque(maxlen=self.arguments["buffer"])

        # if a video path was not supplied, grab the reference
        # to the webcam
        if not self.arguments.get("video", False):
            self.vs = VideoStream(src=0).start()
        else:
            # otherwise, grab a reference to the video file
            self.vs = cv2.VideoCapture(self.arguments["video"])

        # allow the camera or video file to warm up
        time.sleep(self.conf["camera_warmup_time"])

    # detect rectangle acc. to lower / upper
    # num=1: Large Ball, no trace)
    # num=2: Small Ball, with trace)
    def detectRect(self, number, isRect):
        # detect encl. rectangle acc. to lower1 / upper 1 (Large Rectangle)
        # construct a mask for the color number, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        lower = self.lower1 if number == 1 else self.lower2
        upper = self.upper1 if number == 1 else self.upper2
        mask = cv2.inRange(self.hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("Rect-Test", mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the large ball
        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            (center, size, angle) = cv2.minAreaRect(c)
            center = (int(center[0]), int(center[1]))

            # only proceed if the rectangle meets a minimum size
            if (size[0] + size[1]) > self.conf["size1_thresh" if number == 1 else "size2_thresh"]:
                # if rectangle, draw enclosing box
                box = cv2.boxPoints((center, size, angle))
                box = np.int0(box)
                cv2.drawContours(
                    self.frame, [box], 0, self.color1 if number == 1 else self.color2, 2)

                if (number == 1):
                    cv2.putText(self.frame, str(center),
                                (self.frame.shape[1] - 180,
                                 self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 8)
                    cv2.putText(self.frame, str(center),
                                (self.frame.shape[1] - 180,
                                 self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.color1 if number == 0 else self.color1, 2)
                else:
                    cv2.putText(self.frame, str(center), (10, self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 8)
                    cv2.putText(self.frame, str(center), (10, self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.color2, 2)
                return (center)    # required to calculate delta
            return (None)      # no object or too small

    # detect obejct <number> acc. to lower / upper
    # num=1: Large Ball, no trace)
    # num=2: Small Ball, with trace)
    def detectObject(self, number):
        # detect obejct acc. to lower1 / upper 1 (Large Circle, no trace)
        # construct a mask for the color 1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        lower = self.lower1 if number == 1 else self.lower2
        upper = self.upper1 if number == 1 else self.upper2
        mask = cv2.inRange(self.hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("Obj-Test", mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the large ball
        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > self.conf["size1_thresh" if number == 1 else "size2_thresh"]:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(self.frame, (int(x), int(y)), int(radius),
                           self.color1 if number == 1 else self.color2, 7)
                cv2.circle(self.frame, center, 5,
                           self.color1 if number == 1 else self.color2, -1)
                if (number == 1):
                    cv2.putText(self.frame, str(center),
                                (self.frame.shape[1] - 180,
                                 self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 8)
                    cv2.putText(self.frame, str(center),
                                (self.frame.shape[1] - 180,
                                 self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.color1 if number == 0 else self.color1, 2)
                else:
                    cv2.putText(self.frame, str(center), (10, self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 8)
                    cv2.putText(self.frame, str(center), (10, self.frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.color2, 2)
                return (center)  # required to calculate delta
            return (None)  # no object or too small

    # update trace points and draw trace of small ball
    def drawTrace(self, newPoint):
        # update the points queue
        self.pts.appendleft(newPoint)

        # loop over the set of tracked points
        for i in range(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(
                np.sqrt(self.arguments["buffer"] / float(i + 1)) * 2.5)
            cv2.line(self.frame, self.pts[i - 1],
                     self.pts[i], (0, 205, 205), thickness)

    def run(self):
        # print ("RT running: width= ",self.conf["videoframe_width"])
        while not self.event.is_set():
            # grab the current frame
            self.frame = self.vs.read()
            # handle the frame from VideoCapture or VideoStream
            self.frame = self.frame[1] if self.arguments.get(
                "video", False) else self.frame

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if self.frame is None:
                break

            # resize the frame, blur it, and convert it to the HSV
            # color space
            self.frame = imutils.resize(
                self.frame, width=self.conf["videoframe_width"])
            self.blurred = cv2.GaussianBlur(self.frame, (11, 11), 0)
            self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)
            # cv2.imshow("HSV", self.hsv)

            # Detect and draw Large Object: Rectangle or Ball?
            if (self.conf["isRectangle"]):
                center1 = self.detectRect(1, True)
            else:
                center1 = self.detectObject(1)

            # Detect and draw Small Object
            center2 = self.detectObject(2)

            # calculate delta between shapes centers and write to frame
            if (center2 is not None) and (center1 is not None):
                delta = tuple(np.subtract(center2, center1))
                cv2.putText(self.frame, str(delta),
                            (self.frame.shape[1] // 2 -
                             60, self.frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 8)
                cv2.putText(self.frame, str(delta),
                            (self.frame.shape[1] // 2 -
                             60, self.frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
                self.reward.setDeltaIfDifferent(delta)
                # self.reward.print()

            # handle trace-Points and draw Trace of small ball
            self.drawTrace(center2)

            # show the frame to our screen
            cv2.imshow("Frame", self.frame)
            # required to generate video output
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                # Tracker needs to close, then signals to Main-Thread
                self.close()
                os.kill(os.getpid(), signal.SIGINT)
                break
        # run-Loop done, e.g. event.set() in Main-Thread, now closing Camera stream
        # print ("RT stop running")
        self.close()

    # close open stream and open window
    def close(self):
        #print ("RT closing")
        # if we are not using a video file, stop the camera video stream
        if not self.arguments.get("video", False):
            self.vs.stop()
        # otherwise, release the camera
        else:
            self.vs.release()
        time.sleep(0.2)		# needed to avoid rethrown issue... (?)
        # close all windows
        cv2.destroyAllWindows()

        

# end
