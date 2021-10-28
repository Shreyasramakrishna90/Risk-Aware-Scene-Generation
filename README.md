# Risk-Aware Scene Sampling for Dynamic Assurance of Autonomous Systems

ReSonAte uses the information gathered by hazard analysis and assurance cases to build [Bow-Tie Diagrams](https://www.cgerisk.com/knowledgebase/The_bowtie_method) to model hazard propagation paths and capture their relationships with the state of the system and environment. These Bow-tie diagrams are used to synthesize graphical models that are then used at runtime along with the information gathered from prior incidents about the possible environmental hazards and the hypothesis from failure diagnosers and system runtime monitors to estimate the hazard rates at runtime. These hazard rates are then used to determine the likelihood of unsafe system-level consequences captured in the bow-tie diagram. 

This repo has the steps to run the scene generation with the samplers as discussed in the paper. We demonstrate our approach using the CARLA Autonomous Driving Challenge  https://carlachallenge.org/challenge/nhtsa/


## Downloads

1. You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.

2. Download the LEC weights from [here](). 

Save the model.ckpt file to carla-challange/carla_project folder. 

3. Download the trained B-VAE assurance monitor weights from [here](https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/EY5JCqsI65JEtvwMelR6OZwBPfho7FNtBOG5pDWAMXh1ng?e=7hR7pa)

Unzip and save the weights to carla-challange/leaderboard/team_code/detector_code

## Setup Virtual Environment

To run the scene generation workflow with CARLA, clone this repo.

```bash
git clone https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk
```
Then, create a conda environment to run the experiments. 

```
To run this setup first create a virtual environment with python 3.7
conda create -n iccps python=3.7
conda activate py37
cd ${CARLA_ROOT}  # Change ${CARLA_ROOT} for your CARLA root folder
python3 -m pip install -r requirements.txt
```

# Running the Carla setup 

Run the simulator using the following command in one terminal. 

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
In another terminal activate the virtual environment and run the following command.

```bash
conda activate iccps
./run_agent.sh
```

## References

This experiments in this work was build on top of these two works.

```
1. ReSonAte: A Runtime Risk Assessment Framework for Autonomous Systems [paper](https://arxiv.org/abs/2102.09419)

2. Learning By Cheating [paper](https://arxiv.org/abs/1912.12294)

```



