import os
import argparse

import gym
import numpy as np

os.environ['LD_LIBRARY_PATH'] = os.environ['HOME'] + '/.mujoco/mjpro150/bin:'

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.bench import Monitor
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.her.utils import HERGoalEnvWrapper


ENV = "FetchReach-v1"
ALGO = 'ddpg'
NB_TRAIN_EPS = 95000 # corresponding to one HER baseline run for Fetch env
EP_TIMESTEPS = 50
LOG_INTERVAL = 1000 # every 2000 episodes

ALGOS = {
    'sac': SAC,
    'ddpg': DDPG,
    # 'dqn': DQN # does not support continuous actions
}

def find_save_path(dir, trial_id):
    """
    Create a directory to save results and arguments. Adds 100 to the trial id if a directory already exists.

    Params
    ------
    - dir (str)
        Main saving directory
    - trial_id (int)
        Trial identifier
    """
    i=0
    while True:
        save_dir = dir+str(trial_id+i*100)+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i+=1
    return save_dir


def launch(algo, env_id, trial_id, seed):

    logdir = find_save_path('../../data/' + env_id + "/", trial_id)
    # logdir = '/tmp/her/'

    algo_ = ALGOS[algo]
    env = gym.make(env_id)
    # Wrap the environment in a Monitor wrapper to record training progress
    # Note: logdir must exist
    os.makedirs(logdir, exist_ok=True)
    env = Monitor(env, logdir, allow_early_resets=True)

    eval_env = gym.make(env_id)
    eval_env = Monitor(eval_env, logdir, allow_early_resets=True)
    if not isinstance(env, HERGoalEnvWrapper):
        eval_env = HERGoalEnvWrapper(eval_env)

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
                verbose=1, buffer_size=int(1e6), nb_train_steps=100, eval_env=eval_env, nb_eval_steps=20*EP_TIMESTEPS,
                gamma=0.95, batch_size=256,
                policy_kwargs=dict(layers=[256, 256, 256]), **kwargs)

    model.learn(total_timesteps=NB_TRAIN_EPS * EP_TIMESTEPS, log_interval=LOG_INTERVAL)
    model.save(os.path.join(logdir, algo))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', type=str, default=ENV, help='the name of the OpenAI Gym environment that you want to train on')
    parser.add_argument('--trial_id', type=int, default='0', help='trial identifier, name of the saving folder')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1e6), help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--algo', type=str, default=ALGO, help='underlying learning algorithm: td3 or ddpg')
    kwargs = vars(parser.parse_args())
    launch(**kwargs)
