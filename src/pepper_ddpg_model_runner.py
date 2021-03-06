# -*- coding: utf-8 -*-
'''
Ausführen eines trainierten Models
'''
import glob

import tensorflow as tf
import numpy as np
from ddpg.ddpg import getReward, ActorNetwork, \
    OrnsteinUhlenbeckActionNoise
from src.Pepper import Pepper
from src.BallTracker import ballTracker
from src.Pepper.Pepper import readAngle
from src.Settings import *


def main():
    with tf.Session() as sess:
        thread1 = ballTracker.BallTrackerThread()
        thread1.start()
        session = Pepper.init(ip, port)
        Pepper.roboInit(session)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        # critic = CriticNetwork(sess, state_dim, action_dim,
        #                       float(args['critic_lr']), float(args['tau']),
        #                       float(args['gamma']),
        #                       actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        saver = tf.train.Saver()

        models = sorted(glob.glob(args['model'] + "*"))
        print("Model\t\tLowest_Reward\tHighest_Reward\tAvg._Reward")
        for model in models:
            saver.restore(sess, model + "/model")
            print(model + "\t" + runModel(session, thread1, actor, actor_noise))

        print("Finished")
        session.close()
        thread1.exitFlag = 1
        thread1.join()


def runModel(session, thread, actor, actor_noise):
    highestReward = -10000
    lowestReward = 0
    ep_reward = 0
    for j in range(int(args['max_episode_test'])):
        service = session.service("ALMotion")
        params = dict()
        # Hole Anfangszustand
        delta1 = thread.delta[0]
        winkel1 = readAngle(session)
        s = [winkel1, delta1]
        a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
        threshold_reward = 0
        if a[0] < UNTERE_GRENZE:
            a[0] = UNTERE_GRENZE
            threshold_reward = -10000

        if a[0] > OBERE_GRENZE:
            a[0] = OBERE_GRENZE
            threshold_reward = -10000

        # Fuehre Action aus
        params[args['motor']] = [a[0], TIME_TO_MOVE]
        Pepper.move(params, service)

        # Hole Folgezustand
        delta2 = thread.delta[0]

        # Hole Reward
        r = getReward(delta2) + threshold_reward
        if (r > highestReward):
            highestReward = r
        if (r < lowestReward):
            lowestReward = r
        ep_reward += r
    return (str(lowestReward) + "\t\t" + str(highestReward) + "\t\t" + str(ep_reward / int(args['max_episode_len'])))

if __name__ == '__main__':
    main()
