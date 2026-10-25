"""Learning algorithms. All of them train the same Policy network (model.py), use the same
envs and opponents (rollout.py) and are evaluated, logged and checkpointed the same way by
train.py, so runs with different --algo values can be compared directly.

Interface (see Algo): update() collects experience, improves the network once and returns
(number of learner decisions collected, dict of stats for the log).
"""
from .dqn import DQN
from .es import ES
from .pg import A2C, GRPO, PPO, Reinforce

ALGOS = {"ppo": PPO, "a2c": A2C, "reinforce": Reinforce, "grpo": GRPO, "dqn": DQN, "es": ES}
