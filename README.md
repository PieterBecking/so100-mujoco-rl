# Reinforcement Learning for the so100 (so-arm100)
A collection of reinforcement learning experiments for the so100 using MuJoCo Playground.


## Dependencies

This project uses the [Pixi](https://pixi.sh/) package management tool, you will need to install this.

The application uses [MuJoCo Playground](https://playground.mujoco.org/) for reinforcement learning, it is installed as part of the getting started process.


## Getting started

Clone the repo

    git clone https://github.com/lachlanhurst/so100-mujoco-rl.git
    cd so100-mujoco-rl

Install dependencies

    pixi install

The following command will download the so100 MuJoCo model files from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) repo. This may take a little while as it downloads all models, then extracts only those for the so100.

    pixi run download

