import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from TrainAndLoggingCallback import TrainAndLoggingCallback

#setting up game environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#Simplifying the controls from 256 to 7
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#Grayscale the env
env = GrayScaleObservation(env, keep_dim=True)
#Wrapping into dummy env
env = DummyVecEnv([lambda: env])
#Stacking the frames
env = VecFrameStack(env, 4, channels_order='last')

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

print(env.action_space)
print(env.observation_space.shape)
print(SIMPLE_MOVEMENT)

def start_Game():
    done = True
    for frame in range(100):
        if done:
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)
model.save('thisisatestmodel')

# Load model
model = PPO.load('./train/best_model_1000000')
state = env.reset()
# Start the game
state = env.reset()
# Loop through the game
while True:

    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
