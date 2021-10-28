# Risk-Aware Scene Sampling for Dynamic Assurance of Autonomous Systems

This repo has the steps to run the scene generation with the samplers as discussed in the paper. We demonstrate our approach using the CARLA Autonomous Driving Challenge  https://carlachallenge.org/challenge/nhtsa/

## Downloads

1. You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.

2. Download the LEC weights from [here](https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/Eaq1ptU-YJJPrqmEYUK_dx8Bad2KqhVQZJkKwngWnuMWRA?e=U3dtyf). The LEC model architecture was taken from [Learning By Cheating](https://github.com/bradyz/2020_CARLA_challenge)

Save the model.ckpt file to carla-challange/carla_project folder. 

3. Download the trained B-VAE assurance monitor weights from [here](https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/EbB6W8s1XgFJg0Uv762w3v0BuAi7pOrYPZOnbmhHBlEKVQ?e=bOy4Rm)

Unzip and save the weights to carla-challange/leaderboard/team_code/detector_code/ood_detector_weights

## Setup Virtual Environment

To run the scene generation workflow with CARLA, clone this repo.

```bash
git clone https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk
```
Then, create a conda environment to run the experiments. 

```
To run this setup first create a virtual environment with python 3.7
conda create -n iccps python=3.7
cd ${repo}  # Enter into this repo
python3 -m pip install -r requirements.txt
```

# Running the Carla setup 

Run the simulator using the following command in one terminal. 

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=3000 -resx=800 -resy=600 -opengl
```
In another terminal activate the virtual environment and run the following command.

```bash
conda activate iccps
./run_agent.sh
```
This should start running the carla setup with the default random sampler. To select the required sampler check the [sdl](https://github.com/Shreyasramakrishna90/RIsk-Aware-Scene-Generation/tree/main/carla-challange/sdl/scene) repo. 

## References

The experiments in this work was built using these two works.


1. ReSonAte: A Runtime Risk Assessment Framework for Autonomous Systems [paper](https://arxiv.org/abs/2102.09419)

2. Learning By Cheating [paper](https://arxiv.org/abs/1912.12294)




