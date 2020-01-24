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
import tflearn
import argparse
import readAngles
import pepperMove
import json
import ballTracker

from replay_buffer import ReplayBuffer

ip = "192.168.0.40"
port = "9559"
state_dim = 2  # env.observation_space.shape[0]
action_dim = 1  # env.action_space.shape[0]
action_bound = 0.4  # env.action_space.high

TRAINING_STEPS = 50000
OBERE_GRENZE = 0.4
UNTERE_GRENZE = -0.15
TIME_TO_MOVE = 0.3

delta = ""


class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def saveData(data):
    f = open("HipTraining.txt", "a")
    f.write(data)
    f.close()


def readData():
    f = open("HipTraining.txt", "r")
    print(f.read())


def getReward(delta):
    # print("Delta: " + str(delta))
    delta = str(delta).replace("(", "")
    delta = delta.replace(")", "")
    var2_x = delta.partition(",")[0]
    # print("TYPE: " + type(var1_x))
    var2_x = (abs(int(var2_x)))
    reward = 100 - (var2_x)
    return reward


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim  # Winkelmotoren+ kamera
        self.a_dim = action_dim  # Anzahl Motoren z.B. 10
        self.action_bound = action_bound  # Grenze Winkelbewegung z.B. +5
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


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
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        # s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        for j in range(int(args['max_episode_len'])):
            service = session.service("ALMotion")
            params = dict()
            # Hole Anfangszustand
            delta1 = thread.delta[0]
            winkel1 = readAngles.readAngles(session).get("RShoulderPitch")
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
                rewardTMP = -10000

            if a[0] > OBERE_GRENZE:
                # print("Winkel zu gross :" + str(a[0]))
                a[0] = OBERE_GRENZE
                rewardTMP = -10000

            # Fuehre Action aus
            params["RShoulderPitch"] = [a[0], TIME_TO_MOVE]
            pepperMove.move(params, service)

            # Hole Folgezustand
            delta2 = thread.delta[0]
            winkel2 = readAngles.readAngles(session).get("RShoulderPitch")
            s2 = [winkel2, delta2]

            # Hole Reward
            r = getReward(delta2) + rewardTMP
            terminal = False
            # print("Bewegte Motor RShoulderPitch um " + str(a[0]) + " Delta: " + str(delta2) + " " + " Reward: " + str(
            #   r))
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
        print("Epoche: " + str(i) + "\t" + str(ep_reward / 100))
        if i % int(args['save']) == 0 and i != 0:
            print('Saving model')
            saver.save(sess, "model_4/model")


def main(args):
    with tf.Session() as sess:
        print("Starte BallTrackerThread")
        global delta

        thread1 = ballTracker.BallTrackerThread()
        thread1.start()

        print("Main running...")
        session = pepperMove.init(ip, port)
        pepperMove.roboInit(session)

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

        print('The following MODE is detected: %s', args['mode'])
        if args['mode'] == 'INIT':
            saver = tf.train.Saver()
            train(sess, session, thread1, args, actor, critic, actor_noise, False, saver)
        elif args['mode'] == 'TRAIN':
            saver = tf.train.Saver()
            saver.restore(sess, "model_4/model")
            train(sess, session, thread1, args, actor, critic, actor_noise, True, saver)
        # elif args['mode'] == 'TEST':
        #     saver = tf.train.Saver()
        #     saver.restore(sess, "ckpt/model")
        #     testDDPG(sess, args, actor, critic, actor_noise)
        else:
            print('No mode defined!')
        #train(sess, session, thread1, args, actor, critic, actor_noise)

        print('Terminated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)  # 0.0001
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)  # 0.001
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)  # kleineres Gamma
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)  # 0,001
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # Gescheite Trainingsdaten mit Extremwerte ueber den Grenzen
    # Gamme reduzieren
    # Bounding Box der Balltracking normieren

    parser.add_argument('--mode', help='Use INIT, TRAIN', default='TRAIN')
    parser.add_argument('--save', help='how many episodes for saving in INIT and TRAIN', default=1)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=150)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=100)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    main(args)
