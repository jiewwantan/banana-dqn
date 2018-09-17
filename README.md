# Banana Collection in Unity environment
Banana collection navigation by a vanilla DQN agent

![Trained_agent](trained_agent.gif)

## Environment Details
The objective of this project is to collect as many yellow bananas as possible while avoiding the blue bananas. 
The following is the environment where the agent performs its task. 

<mark>Unity Academy name: Academy  
        Number of Brains: 1  
        Number of External Brains : 1  
        Lesson number : 0  
        Reset Parameters :  </mark>  

<mark>  Unity brain name: BananaBrain  
        Number of Visual Observations (per agent): 0  
        Vector Observation space type: continuous  
        Vector Observation space size (per agent): 37  
        Number of stacked Vector Observation: 1  
        Vector Action space type: discrete  
        Vector Action space size (per agent): 4  
        Vector Action descriptions: , , , </mark>  

The environment has 37 state spaces for the agent to operate in. There are 4 discrete actions spaces for forward, backward, turn left, turn right moves. The agent's problem is considered solved when average score of 13 is attained. The agent aim to achieve within 1800 episodes. 

## Installation Instruction
#### The README has instructions for installing dependencies or downloading needed files.

Python 3.6 is required. The program requires PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

```git clone https://github.com/udacity/deep-reinforcement-learning.git  
cd deep-reinforcement-learning/python  
pip install .```

Run the following to create drlnd kernel in ipython so that the right unity environment is loaded correctly  
<mark>python -m ipykernel install --user --name drlnd --display-name "drlnd"</mark>



## Getting Started

Place <mark>report.ipynb</mark> in the folder <mark>p1_navigation/</mark> together with the following two files:

1. dqn_agent.py - contains the DQN agent code. 
2. model.py - contains neural network class for used as Q function

The Unity Banana collection environment can be downloaded from here: 

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)  
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)  
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)  
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)  

Choose the environment suitable for your machine. Unzipping will create another Banana_xxxx folder. For example, if the Linux Banana environment is downloaded, <mark>Banana_Linux</mark> will be created. 

Run p1_navigation/report.ipynb

Enter the right path for the Unity Banan environment in report.ipynb. For example for a 64-bit Linux machine: 
env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64")

Run the remaining cell as ordered in report.ipynb to train the DQN agent. 




```python

```
