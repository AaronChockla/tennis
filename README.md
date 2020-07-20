# Train Two RL Agents to Play Tennis
## Udacity Deep Reinforcement Learning Nanodegree Project

### Environment
<a target="_blank" rel="noopener noreferrer" href="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif"><img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif" alt="Trained Agent" title="Trained Agent" style="max-width:100%;"></a>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single score for each episode.
 - The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
 
### Getting Started
1. Clone this repository.

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:

 - **Version 1:** Single Agent

    - **Linux:** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip">click here</a>
    - **Mac OSX:** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip">click here</a>
    - **Windows (32-bit):** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip">click here</a>
    - **Windows (64-bit):** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip">click here</a>

- **Version 2:** Distributed Agents (20)

    - **Linux:** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip">click here</a>
    - **Mac OSX:** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip">click here</a>
    - **Windows (32-bit):** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip">click here</a>
    - **Windows (64-bit):** <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip">click here</a>

 - **AWS:** If you'd like to train the agent on AWS (and have not enabled a virtual screen), use <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip">this link</a> (version 1) or <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip">this link</a> (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

3. Place the downloaded file(s) in the folder you cloned this repo to and unzip (or decompress) the file.

4. Create a Python environment for this project, *e.g.*, using conda or venv.

5. Activate that environment and install dependencies: <code>pip install -r requirements.txt</code>

### Instructions
Follow the instructions in Tennis.ipynb to get started with training your own agent. The code is documented. Model weights are stored in <code>checkpoint.pth</code> files
