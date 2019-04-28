import os
import argparse

import gym
import numpy as np

os.environ['LD_LIBRARY_PATH'] = os.environ['HOME'] + '/.mujoco/mjpro150/bin:'

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.bench import Monitor
from stable_baselines.ddpg import NormalActionNoise
# from custom version of stable-baselines ?
from stable_baselines.her.utils import find_save_path

ENV = "FetchPush-v1"
NB_EPOCHS = 1000
ALGO = 'sac'
TIMESTEPS_PER_EPOCH = 100  #TODO: check how many timesteps per epoch

ALGOS = {
    'sac': SAC,
    'ddpg': DDPG,
    # 'dqn': DQN # does not support continuous actions
}

def launch(algo, env, trial_id, n_epochs, seed):

    logdir = find_save_path('../../data/' + env + "/", trial_id)
    # logdir = '/tmp/her/'

    algo_ = ALGOS[algo]
    env = gym.make(env)
    # Wrap the environment in a Monitor wrapper to record training progress
    # Note: logdir must exist
    os.makedirs(logdir, exist_ok=True)
    env = Monitor(env, logdir, allow_early_resets=True)

    if algo_ == SAC:
        kwargs = {'learning_rate': 1e-3}
    elif algo_ == DDPG:
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

        kwargs = {
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'action_noise': action_noise
        }
    else:
        raise ValueError('Algo not supported: {}'.format(algo_))

    model = HER('MlpPolicy', env, algo_, n_sampled_goal=4, goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e6),
                gamma=0.95, batch_size=256,
                policy_kwargs=dict(layers=[256, 256, 256]), **kwargs)

    model.learn(total_timesteps=n_epochs * TIMESTEPS_PER_EPOCH)
    model.save(os.path.join(logdir, algo))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default=ENV, help='the name of the OpenAI Gym environment that you want to train on')
    parser.add_argument('--trial_id', type=int, default='0', help='trial identifier, name of the saving folder')
    parser.add_argument('--n_epochs', type=int, default=NB_EPOCHS, help='the number of training epochs to run')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1e6), help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--algo', type=str, default=ALGO, help='underlying learning algorithm: td3 or ddpg')
    kwargs = vars(parser.parse_args())
    launch(**kwargs)
