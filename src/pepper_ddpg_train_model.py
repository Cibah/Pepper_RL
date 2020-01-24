""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import json
import argparse
import pprint as pp
from ddpg.ddpg import ActorNetwork, CriticNetwork, OrnsteinUhlenbeckActionNoise, testDDPG
from replay_buffer import ReplayBuffer
from src.ddpg.ddpg import build_summaries

filename = "model_5"

def train(sess, args, actor, critic, actor_noise, update_model, saver):
    
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
    # tflearn.is_training(True)

    TIME_TO_MOVE = 0.3
    MULTIPLICATOR = 0
    STEPSIZE = 100

    for i in range(int(args['max_episodes'])):
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):


            with open('Pepper_Training.txt') as json_file:
                data = json.load(json_file)
                q = data['steps']

                #p = q[j + (STEPSIZE * MULTIPLICATOR)]
                p = q[j]
                s = [p['az'], p['ad']]
                # Kombiniere az mit ad ?
                # s = az + ad

                # s2 = fz + fd

                a = p['actionR']

                # actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
                s2 = [p['fz'], p['fd']]
                r = p['rw']
                # info = False Wird nicht benutzt??
                terminal = False
                replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

            #a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            #s2, r, terminal, info = env.step(a[0])

            #replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

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
            #print('Step completed: ' + str(j))
            ep_reward += r
        #MULTIPLICATOR = MULTIPLICATOR + 1
        print("Epoche: " + str(i) + "\t" + str(ep_reward / int(args['max_episode_len'])))
        if i % int(args['save']) == 0 and i != 0:
            print('Saving model')
            saver.save(sess, filename)
    

        

def main(args):

    with tf.Session() as sess:

        #env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        #env.seed(int(args['random_seed']))

        state_dim = 2 #env.observation_space.shape[0]
        action_dim = 1 #env.action_space.shape[0]
        action_bound = 0.46 #env.action_space.high
        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # if args['use_gym_monitor']:
        #     if not args['render_env']:
        #         env = wrappers.Monitor(
        #             env, args['monitor_dir'], video_callable=False, force=True)
        #     else:
        #         env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        print('The following MODE is detected: %s', args['mode'])
        if args['mode'] == 'INIT':
            saver = tf.train.Saver()
            train(sess, args, actor, critic, actor_noise, False, saver)
        elif args['mode'] == 'TRAIN':
            saver = tf.train.Saver()
            saver.restore(sess, filename + "/model")
            train(sess, args, actor, critic, actor_noise, True, saver)
        elif args['mode'] == 'TEST':
            saver = tf.train.Saver()
            saver.restore(sess, filename + "/model")
            testDDPG(sess, args, actor, critic, actor_noise)
        else:
            print('No mode defined!')


        print('Terminated')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.01)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.01)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    parser.add_argument('--mode', help='Use INIT, TRAIN or TEST', default='INIT')
    parser.add_argument('--save', help='how many episodes for saving in INIT and TRAIN', default=20)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1500)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=2500)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
