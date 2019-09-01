import argparse
import datetime
import sys
import robosumo
import gym
import multiprocessing
import os.path as osp
import tensorflow as tf
from baselines import logger
from alg import learn
from defaults import get_default_params
from sumo_env import SumoEnv
from subproc_vec_env import SubprocVecEnv
from baselines.common.tf_util import get_session


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def make_env_from_id(env_id, seed, prefix):
    env = gym.make(env_id)
    env = SumoEnv(env, allow_early_resets=True, file_prefix=prefix)
    env.seed(seed)
    return env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    print('num of env: ' + str(nenv))

    seed = args.seed
    env_id = args.env

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    env = SubprocVecEnv([lambda: make_env_from_id(env_id, seed + i if seed is not None else None, "")
                         for i in range(nenv)])
    return env


def train(args, extra_args):
    assert args.env[:8] == 'RoboSumo'

    env_id = args.env

    # build a temporary environment to get number of agents
    temp_env = gym.make(env_id)
    nagent = len(temp_env.agents)
    temp_env.close()

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_default_params()
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = 'mlp'

    print('Training PPO2 on {} with arguments \n{}'.format(env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        nagent=nagent,
        **alg_kwargs
    )

    return model, env


def main(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel.', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)

    args, unknown_args = parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.log_path is not None:
        args.log_path = osp.join(args.log_path,
                                 args.env + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
    else:
        save_path = osp.join(args.log_path, 'model')
    print('Saving final model to', save_path)
    model.save(save_path)

    env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)
