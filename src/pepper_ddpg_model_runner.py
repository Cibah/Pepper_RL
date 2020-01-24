# -*- coding: utf-8 -*-
"""
Ausfuehren eines trainierten Models
"""
import tensorflow as tf
import numpy as np
from ddpg.ddpg import build_summaries, getReward, ReplayBuffer, ActorNetwork, CriticNetwork, \
    OrnsteinUhlenbeckActionNoise
from src.Pepper import Pepper
from src.BallTracker import ballTracker
from src.Pepper.Pepper import readAngles
from src.Settings import *


def main():
    with tf.Session() as sess:
        global delta

        thread1 = ballTracker.BallTrackerThread()
        thread1.start()
        session = Pepper.init(ip, port)
        Pepper.roboInit(session)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        saver = tf.train.Saver()
        saver.restore(sess, args['model'])
        testDDPG(session, thread1, actor, critic, actor_noise)
        print('Terminated')


def testDDPG(session, thread, actor, critic, actor_noise):
    # test for max_episodes number of episodes
    for i in range(int(args['max_episodes'])):

        ep_reward = 0
        for j in range(int(args['max_episode_len'])):
            service = session.service("ALMotion")
            params = dict()
            # Hole Anfangszustand
            delta1 = thread.delta[0]
            winkel1 = readAngles(session).get(args['motor'])
            s = [winkel1, delta1]
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            rewardTMP = 0
            if a[0] < UNTERE_GRENZE:
                # print("Winkel zu klein :" + str(a[0]))
                a[0] = UNTERE_GRENZE
                rewardTMP = -1000

            if a[0] > OBERE_GRENZE:
                # print("Winkel zu gross :" + str(a[0]))
                a[0] = OBERE_GRENZE
                rewardTMP = -1000

            # Fuehre Action aus
            params[args['motor']] = [a[0], TIME_TO_MOVE]
            Pepper.move(params, service)

            # Hole Folgezustand
            delta2 = thread.delta[0]

            # Hole Reward
            r = getReward(delta2) + rewardTMP
            terminal = False

            ep_reward += r

            if terminal:
                print('| Episode: {:d} | Reward: {:d} |'.format(i, int(ep_reward)))
                break
    print("Epoche: " + str(i) + "\t" + str(ep_reward / int(args['max_episode_len'])))


if __name__ == '__main__':
    main()
