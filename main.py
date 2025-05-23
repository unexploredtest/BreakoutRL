import numpy as np
import torch
from PIL import Image
import gymnasium as gym
from ale_py import ALEInterface
ale = ALEInterface()

from breakout import DQNBreakout
from model import AtariNet
from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = DQNBreakout(device=device)

model = AtariNet(nb_actions=4)

model = model.to(device)

model.load_the_model()

agent = Agent(model=model, device=device, epsilon=1.0, nb_warmups=5000, nb_actions=4,
    learning_rate=0.00001, memory_capacity=1000000, batch_size=64)


agent.train(env=env, epochs=200000) # The youtuber PC took 32 secs for 10 epochs


