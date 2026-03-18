# 🤖 Hands-on Reinforcement Learning

A practical repository for learning **Reinforcement Learning (RL)** from scratch through **implementation, experimentation, and analysis**.

Unlike many resources that focus heavily on theory, this project emphasizes **building algorithms from scratch**, understanding their mechanics, and applying them to small environments and mini-projects.

The goal is to help you move from **basic RL concepts → modern deep RL implementations**.

---
## ❗ Prerequesits
If you are a beginner with no prior experience in Python, machine learning fundamentals, or deep neural networks, it is highly recommended to first build a foundation in these areas before diving into reinforcement learning implementations.

Helpful background knowledge includes:
- python programming
- Linear algebra and probability basics
- Machine learning fundamentals
- Neural networks and deep learning (preferably with PyTorch or TensorFlow)

Having these prerequisites will make it significantly easier to understand the algorithms and implementations provided in this repository.

---
## 🙏 Acknowledgement

This repository was inspired by several excellent educational resources.

The primary inspiration comes from the lecture series by WINDY Lab and Shiyu Zhao, which provides a very clear and beginner-friendly introduction to reinforcement learning:

- https://www.youtube.com/watch?v=ZHMWHr9811U&list=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8

Additional inspiration was taken from the following reinforcement learning playlist:

- https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
- https://lilianweng.github.io/posts/2018-02-19-rl-overview/

Special thanks to the creators of these resources for making high-quality reinforcement learning education accessible to everyone.

---

## 📖 What You'll Learn

In this repository, I've implemented several popular reinforcement learning algorithms and environments **from scratch** to help build a deeper understanding of how they work internally.

To keep the learning process consistent and easier to follow, most of the algorithms are implemented using variations of the **Frozen Lake environment**. The environment is gradually modified based on algorithm requirements. I believe sticking to a single environment makes it easier to understand how different RL algorithms behave and how their implementations differ.

In the **mini-projects** section, additional reinforcement learning problems will be introduced. These projects will focus more on building RL systems using **popular libraries and frameworks**, providing a bridge between educational implementations and **industry-oriented applications**.

>TL;DR:
>- Build a **deep understanding of RL algorithms**
>- Implement **algorithms from scratch**
>- Provide **clean educational implementations**
>- Bridge **classical RL → deep reinforcement learning**


## 🧠 Algorithms Implemented

### 1️⃣ Foundations ✅
- Reinforcement Learning basics
- Markov Decision Processes (MDPs)
- NxM grid world

### 2️⃣ Model-Based Methods (Dynamic Programming) ✅

- Policy Iteration
- Value Iteration
- Truncated Policy Iteration

### 3️⃣ Monte Carlo Methods ✅

- Monte Carlo (Basic)
- Monte Carlo with Exploring Starts
- Monte Carlo with **$\epsilon$-greedy policy**

### 4️⃣ Stochastic Approximation ✅
- Theoretical foundations and analysis of algorithm behind incremental updates used in RL algorithms

### 5️⃣ Temporal Difference Learning ✅
- SARSA
- Q-Learning (Off-policy)
- Q-Learning (On-policy)

### 6️⃣ Value Function Approximation ✅
- SARSA with Value Function Approximation
- Q-Learning with Value Function Approximation
- Deep Q-Network (DQN)

### 7️⃣ Policy Gradient Methods ✅
- REINFORCE algorithm using simple linear function from scratch
- REINFORCE algorithm using NN in pytorch

### 8️⃣ Actor-Critic Methods ⚠️
- simple Actor-critic(QAC) method from scratch ✅
- Simple Actor-Critic (QAC) using NN in pytorch  ❌
- Advantage Actor-Critic (A2C)  ❌

### Status Legend

- ✅ **Completed** — Implementation is finished and no issues have been identified.
- ⚠️ **Needs Review** — The implementation works but may contain potential issues and will be reviewed soon.
- ❌ **Incorrect** — The implementation is known to be incorrect and will be reviewed and fixed soon.



## 🎮 Mini Projects

- Tic-Tac-Toe with Deep Q-Learning


## 🛠 Tech Stack

* Python == 3.10.14
* NumPy == 2.2.5
* PyTorch == 2.5.1
* Matplotlib == 3.9.2
* Pandas == 2.2.3
* Jupyter Notebook



## 📈 Future Additions

Planned extensions:

* PPO (Proximal Policy Optimization)
* DDPG
* Soft Actor-Critic (SAC)
* Multi-Agent RL
* More RL environments


## 🤝 Contributions

Contributions, improvements, and suggestions are welcome.

If you find issues or have ideas for improvements, feel free to open an issue or submit a pull request.



## 📜 License

This repository is open-source and available under the **MIT License**.


## 📬 Contact

For any inquiries, suggestions, or feedback, feel free to reach out.

[![Email](https://img.shields.io/badge/Email-dhakalsumn739%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:dhakalsumn739@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Suman%20Dhakal-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/suman-dhakal-2822a1198/)


## References
- https://www.youtube.com/watch?v=ZHMWHr9811Ulist=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8&index=3
- https://www.youtube.com/playlistlist=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
- https://mhahsler.github.io/Introduction_to_Reinforcement_Learning/
- http://gibberblot.github.io/rl-notes/intro/intro.html
- https://lilianweng.github.io/posts/2018-02-19-rl-overview/
