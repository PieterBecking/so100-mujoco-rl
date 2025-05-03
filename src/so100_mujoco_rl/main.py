import click
import gymnasium as gym
import logging
import numpy as np
import os
import stable_baselines3

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# while not called directly, we need to import this so the environments are registered
import so100_mujoco_rl

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create directories to hold models, logs, sim videos
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "movies"


def algorithm_factory(algorithm_name: str, env: gym.Env) -> BaseAlgorithm:
    """
    Factory method to create algorithm. Will also include any extra params needed for this
    particular case (balancing robot)
    """
    if algorithm_name == "DDPG":
        policy_kwargs = dict(
            net_arch=dict(pi=[300, 200], qf=[200, 150])
        )
        action_noise = NormalActionNoise(
            mean=np.zeros(2),
            sigma=0.1 * np.ones(2)
        )
        model = stable_baselines3.DDPG(
            "MlpPolicy",
            env=env,
            verbose=1,
            device='cpu',
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            action_noise=action_noise
        )
        return model
    elif algorithm_name == "PPO":
        model = stable_baselines3.PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            device='cpu',
            tensorboard_log=LOG_DIR
        )
        return model
    else:
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            return None
        model = algorithm_class(
            'MlpPolicy',
            env,
            verbose=1,
            device='cpu',
            tensorboard_log=LOG_DIR
        )
        return model

@click.command(name="test", help="Test the current model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.option('--show-i', is_flag=True, default=False, help="log model inputs to std out in Python array syntax")
@click.pass_context
def test(ctx: dict, environment: str, show_io: bool, show_i: bool):
    """ Test a model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
    model = algorithm_class.load(model_file, env=env)

    run_loop_count = 0
    terminated_loop_count = 0
    obs = env.reset()[0]
    while True:
        action, _ = model.predict(obs)
        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(action)))
        if show_i and run_loop_count % 30 == 0:
            logger.info(str(list(obs)) + ",")

        obs, _, terminated, truncated, _ = env.step(action)

        if (terminated or truncated) and terminated_loop_count > 200:
            terminated_loop_count = 0
            obs = env.reset()[0]
            # break
        if terminated or truncated:
            terminated_loop_count +=1

        run_loop_count += 1


@click.command(name="record", help="Record a model with a given environment")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def record(ctx: dict, environment: str):
    vec_env = DummyVecEnv([lambda: gym.make(environment, render_mode="rgb_array")])

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting test simulation for recording")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
    model = algorithm_class.load(model_file, env=vec_env)

    video_length = 3000

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(
        vec_env,
        RECORDING_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"rec-{environment}"
    )

    obs = vec_env.reset()
    vec_env.reset()

    with click.progressbar(length=video_length + 1, label="Recording video") as bar:
        for _ in bar:
            action, _ = model.predict(obs)
            obs, _, terminated, truncated = vec_env.step(action)

    # Save the video
    vec_env.close()


@click.command(name="train", help="Train a model with a given environment")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def train(ctx: dict, environment: str):
    """ Train a model by with a given MuJoCo environment """

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    env = gym.make(environment, render_mode='rgb_array')
    env = Monitor(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=RECORDING_DIR,
        video_length=0,
        episode_trigger = lambda x: x % 50 == 0,
    )

    logger.info(f"Starting training process")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # no model given, so create a new one
        # model = algorithm_class('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=LOG_DIR)
        model = algorithm_factory(algorithm_name=algorithm_name, env=env)
        logger.info(f"Model: starting with new model")
    elif os.path.isfile(model_file):
        # start with an existing model
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            raise RuntimeError(f"Couldn't find algorithm {algorithm_name}")
        model = algorithm_class.load(model_file, env=env, tensorboard_log=LOG_DIR)
        logger.info(f"Model: starting with {model_file}")
    else:
        raise RuntimeError(f"Model file {model_file} does not exist")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6000, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10000,
        verbose=1
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=20000,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        verbose=1,
        best_model_save_path=os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=40000,
        verbose=2,
        save_path=os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}"),
        name_prefix=f"{environment}_{algorithm_name}_cp_"
    )

    model.learn(
        total_timesteps=int(1e10),
        tb_log_name=f"{environment}_{algorithm_name}",
        callback=CallbackList([checkpoint_callback, eval_callback])
    )


@click.group()
@click.option(
    '-a',
    '--algorithm',
    required=True,
    type=str,
    default='PPO',
    help="Stable Baseline3 algorithm (eg; A2C, DDPG, DQN, PPO, SAC, TD3)"
)
@click.option(
    '-m',
    '--model',
    default=None,
    type=click.Path(exists=False),
    help="Path to model file"
)
@click.pass_context
def cli(ctx: dict, algorithm: str, model: str | os.PathLike):
    algorithm_class = getattr(stable_baselines3, algorithm, None)
    if algorithm_class is None:
        raise RuntimeError(f"Could not find Stable Baselines3 algorithm for: {algorithm}")

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['ALGORITHM_NAME'] = algorithm
    ctx.obj['MODEL_PATH'] = model


cli.add_command(record)
cli.add_command(test)
cli.add_command(train)


def __make_folders():
    # create folders to store training data, models, logs, etc
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)


if __name__ == '__main__':
    __make_folders()
    cli(obj={})
