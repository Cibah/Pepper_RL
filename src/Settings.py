# -*- coding: utf-8 -*-
ip = "192.168.0.40"
port = "9559"
state_dim = 2  # env.observation_space.shape[0]
action_dim = 1  # env.action_space.shape[0]
action_bound = 0.4  # env.action_space.high
TRAINING_STEPS = 2000
OBERE_GRENZE = 0.4
UNTERE_GRENZE = -0.25
TIME_TO_MOVE = 0.3
TRAINING_FILE = "training_set.json"
VIDEODEVICE = "/dev/video1"
BALLTRACKERCONFIG = "conf.json"

delta = ""

args = dict()
args['model'] = "feedback_4"
args['motor'] = "RShoulderPitch"
args['summary_dir'] = "./results/tf_ddpg"
args['buffer_size'] = 1000000
args['random_seed'] = 1234
args['max_episodes'] = 10000
args['max_episode_len'] = 1000
args['max_episode_test'] = 1000
args['minibatch_size'] = 64
args['actor_lr'] = 0.0001
args['critic_lr'] = 0.001
args['gamma'] = 0.2
args['tau'] = 0.001
args['env'] = 'Pendulum-v0'
args['render_env'] = 0.001
args['monitor_dir'] = './results/gym_ddpg'
args['save'] = 200
