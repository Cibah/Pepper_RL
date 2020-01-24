# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import numpy as np
import tflearn

from ddpg.ddpg import build_summaries, ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise, ReplayBuffer, \
    getReward

from src.BallTracker import ballTracker
from src.Pepper import Pepper
from src.Pepper.Pepper import readAngles
from src.Settings import *


# ===========================
#   Agent Training
# ===========================

def train(sess, session, thread, args, actor, critic, actor_noise, update_model, saver):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    if update_model == False:
        sess.run(tf.global_variables_initializer())
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()


    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        # s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(int(args['max_episode_len'])):
            service = session.service("ALMotion")
            params = dict()
            # Hole Anfangszustand
            delta1 = thread.delta[0]
            winkel1 = readAngles(session).get(args['motor'])
            s = [winkel1, delta1]

            # Hole action
            a = actor.predict(np.reshape(s, (1, 2))) + (1. / (1. + i))
            # ITERATE THORUGH SAMPLED DATA AND ADD TO REPLAY BUFFER

            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            # a[0] = ((a[0] - (1 - action_bound)) / (action_bound - (1 - action_bound))) * (
            #            OBERE_GRENZE - UNTERE_GRENZE) + UNTERE_GRENZE
            # a[0] = a[0]
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
            winkel2 = readAngles(session).get(args['motor'])
            s2 = [winkel2, delta2]

            # Hole Reward
            r = getReward(delta2) + rewardTMP
            terminal = False
            print(str(a[0]) + "\t" + str(
                r))

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # export actor Model somewhere

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, (ep_ave_max_q / float(j))))
                break


def main():
    with tf.Session() as sess:
        print("Starte BallTrackerThread")
        global delta

        thread1 = ballTracker.BallTrackerThread()
        thread1.start()

        print("Main running...")
        session = Pepper.init(ip, port)
        Pepper.roboInit(session)

        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['mode'] == 'INIT':
            saver = tf.train.Saver()
            train(sess, args, actor, critic, actor_noise, False, saver)
        elif args['mode'] == 'TRAIN':
            saver = tf.train.Saver()
            saver.restore(sess, args['model'] + "/model")
            train(sess, args, actor, critic, actor_noise, True, saver)
        elif args['mode'] == 'TEST':
            saver = tf.train.Saver()
            saver.restore(sess, args['model'] + "/model")
            testDDPG(sess, args, actor, critic, actor_noise)
        else:
            print('No mode defined!')
        #train(sess, session, thread1, args, actor, critic, actor_noise)
        print('Terminated')


if __name__ == '__main__':
    main()
