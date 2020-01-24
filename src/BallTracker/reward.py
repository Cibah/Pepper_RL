# reward.py:
# Klasse, um Reward-Daten zu erfassen und weiterzugeben
# wird importiert von der Main-Appliukation und der rewardTracker Class
#
#
import threading  # required for lock


class Reward:
    def __init__(self):
        self._lock = threading.Lock()
        self.isNew = False		# True, iff new Value is written
        self.delta = (0.0, 0.0,)  # vector to indicate difference

    # set new Delta, if different, set isNew to True
    def setDeltaIfDifferent(self, newDelta):
        with self._lock:
            if self.delta != newDelta:
                self.delta = newDelta
                self.isNew = True
        return

    # returns delta if new, otherwise returns False, sets isNew to False
    def getDeltaIfNew(self):
        with self._lock:
            if self.isNew:
                self.isNew = False
                return self.delta
            else:
                return None


# das ist schon alles
