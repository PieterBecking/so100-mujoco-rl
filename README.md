# Reinforcement Learning for the so100 (so-arm100)
A collection of reinforcement learning experiments for the so100 using MuJoCo, Gymnasium, and Stable Baselines3.


## Dependencies

This project uses the [Pixi](https://pixi.sh/) package management tool, you will need to install this.

The following dependencies are used for simulation and RL, all of which are installed as part of the getting started process.
- [MuJoCo](https://mujoco.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) 


## Getting started

Clone the repo

    git clone https://github.com/lachlanhurst/so100-mujoco-rl.git
    cd so100-mujoco-rl

Install dependencies

    pixi install

The following command will download the so100 MuJoCo model files from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) repo. This may take a little while as it downloads all models, then extracts only those for the so100.

    pixi run download

