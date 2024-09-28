[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Train car to run on city map using T3D. 
- Record your best attempt (it will be evaluated on how good your car's driving was in this video).
- Upload the video on YouTube and share the link

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]

## :open_file_folder: Files
- [**ai.py**](ai.py)
    - This file contains class for model and dqn
- [**map.py**](map.py)
    - This is the main file of this project
    - It uses function available in `ai.py`
    - It contains functions to move object, calculate reward and display.

## :building_construction: Implementation Details
The code is implemented based on Addressing Function Approximation Error in Actor-Critic Methods
paper. The T3D builds on Double Q-learning, by taking the minimum value between a pair of critics to
limit over estimation. It delays the policy update to reduce per-update error and further improve 
performance. 

## :flight_departure: Implementation Steps
### Step 1: Defining Replay memory:

A replay memory holds the experience of an agent while traversing through the environment. It
observes the agents action and stores these observations. At any time step an observation can be
defined by the agent's current state, next state, action it takes in the current state to move to
next state and reward it earns from this step. Each observation is called as transition.

We define a class for Replay memory. Each object of replay memory has a storage array of max_size N
(1e6 in my case). The storage array is filled up with tuples of transition first completely randomly
based on random action steps of the agent, but as the agent keeps learning the environment, we fill
the replay buffer with actions predicted by our RL model. A transition tuple also saves a done flag,
a Boolean value which is set as 1 if the episode terminates after the action step.

### Step 2: Defining the Actor Model architecture:

It decides which action to take. It takes state as input. It essentially controls how the agent
behaves by learning the optimal policy (policy-based). 

### Step 2: Defining Two Critic Model architecture:

Critic evaluates the action predicted by actor by computing the value function.

### Step 4: Sampling Transitions

Sample from a batch of transitions from the Replay memory storage

### Step 5: Actor Target Predicts Next Action 

The actor target network uses the next state from the transition s' to predict the next action a'.
It uses the forward() in actor class for prediction. 

### Step 6: Noise regularization on the predicted next action a'

Before sending a' to critic target networks, we add Gaussian noise to this next action a' and we
clamp it in a range of values supported by the environment. So if we maximize our value estimates
over actions with noise, we can expect our policies to be more stable and robust. It also introduces
some sort of exploration to our agent.

### Step 7: Q Value Estimation by Critic Targets

Predict Q values from both Critic target and take the minimum value

Both Critic targets take (s', a') as input and return Q values, Qt1(s', a') and Qt2(s', a') as
outputs.

### Step 8: Target value Computation

We use the target_Q computed in the last code block in the Bellman's equation as below:

$$
\begin{align*}
Qt = r + \gamma * min(Qt1, Qt2)
\end{align*}
$$

### Step 9: Q value Estimation by Critic Models

Two critic models take (s, a) and return two Q-Values

### Step 10: Compute the Critic loss 

We compute the critic loss using the Q-values returned from the Critic model networks.

$$
Critic\ Loss = MSE(Q1(s,a),Qt) + MSE(Q2(s,a),Qt)
$$

### Step 11: Update Critic Models

Backpropagate using Critic Loss and update the parameters of two Critic models.

### Step 12: Update Actor Model

Once every two iterations, we update our Actor Model by performing gradient ascent on the output of the first Critic Model.

### Step 13: Update Actor Target

We soft update our actor target network using Polyak averaging. It is delayed and done after every two actor model update.

Polyak Averaging: 

$$
\theta' = \tau\theta + (1-\tau)\theta
$$


This way our target comes closer to our model.

### Step 14: Update Critic Target 

We soft update our critic target network along with our Actor Target using Polyak averaging.

$$
\phi' = \tau \phi + (1-\tau)\phi'
$$

## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/map_explorer_T3D
```
2. Go inside folder
```
 cd map_explorer_T3D
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start exploring with:
python map.py

```

## :video_camera: Demo
[Exploring SVNIT Campus with AI: Twin Delayed DDPG in Action](https://www.youtube.com/watch?v=ON3reORkCkU)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/