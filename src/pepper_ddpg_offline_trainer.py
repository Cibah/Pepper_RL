# -*- coding: utf-8 -*-
"""
Pepper Train offline a Model with given training_sets.
"""
import json
import random

import tensorflow as tf
import numpy as np

from src.Settings import *
from ddpg.ddpg import ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise, ReplayBuffer, build_summaries


def trainFromDataset(sess, args, actor, critic, actor_noise, update_model, saver):
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
    file = open(TRAINING_FILE)
    datasets = json.load(file)['steps']
    file.close()

    for i in range(int(args['max_episodes'])):
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            randomnumber = random.randint(0, len(datasets) - 1)
            p = datasets[randomnumber]
            s = [p['az'], p['ad']]
            a = p['actionR']
            s2 = [p['fz'], p['fd']]
            r = p['rw']
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              False, np.reshape(s2, (actor.s_dim,)))

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

            ep_reward += r
        print("Episode: " + str(i) + "\t" + str(ep_reward / args['max_episode_len']) + "\tbeendet")
        if i % int(args['save']) == 0 and i != 0:
            print('Saving model')
            saver.save(sess, args['model'] + "_" + i + "/model")


def main():
    print("Doing Offline Training")
    with tf.Session() as sess:
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        saver = tf.train.Saver()
        trainFromDataset(sess, args, actor, critic, actor_noise, False, saver)


if __name__ == '__main__':
    main()
