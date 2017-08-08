
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras import backend as K 


EPISODES = 100000
MEM_LEN = 1000000
EXPLORE_FRAMES = 50000
DECAY_FRAMES = 1000000
TOTAL_FRAMES = 10000000
TARGET_UPDATE = 10000

GAME = 'PongNoFrameskip-v4'
SAVE_DIR = '/data/ssd/public/weixiao/save_model_2/'
# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def __init__(self, state_size, action_size):
        # if you want to see Pong learning, then change to True
        self.render = 0
        self.load_model = 0

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # use consecutive 4 states for learning 
        self.current_state = np.zeros((1,state_size[0],state_size[1],4))
        

        # These are hyper parameters for the DQN
        self.current_frame = 0
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay = (self.epsilon-self.epsilon_min)/DECAY_FRAMES
        self.batch_size = 32
        self.train_start = EXPLORE_FRAMES
        # create replay memory using deque
        self.memory = deque(maxlen=MEM_LEN)

        # create main model
        self.model = self.build_model()
        self.target_model = self.build_model()

        if self.load_model:
            self.model.load_weights(SAVE_DIR+GAME+".h5")
        #result
        self.loss = -1
        self.Q = -1

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=4, padding='valid', kernel_initializer = 'he_uniform', input_shape=((self.state_size[0],self.state_size[1],4)), activation='relu'))
        model.add(Conv2D(32, (5, 5), strides=2, padding='valid', kernel_initializer = 'he_uniform', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        # Compile model        
        rms = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss=self._huber_loss, optimizer=rms, metrics=['accuracy'])
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self):
        self.train_model()
        if np.random.rand() <= self.epsilon or self.current_frame < EXPLORE_FRAMES:
            return random.randrange(self.action_size)
        else: 
            q_value = self.model.predict(np.float32(self.current_state)/255.)
            self.Q = np.mean(q_value[0])
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, action, reward, next_state, done):
        new_state = np.append(self.current_state[...,1:],np.reshape(next_state,[1,self.state_size[0],self.state_size[1],1]),axis = 3)
        self.memory.append((self.current_state, action, reward, new_state, done))
        self.current_state = new_state
        #update time
        self.current_frame += 1
        # for every 10000 frame
        if self.current_frame%TARGET_UPDATE == 0:
            #update target model
            self.update_target_model()
            #save model
            agent.model.save_weights(SAVE_DIR+GAME+".h5")
        # update epsilon
        if self.current_frame > EXPLORE_FRAMES and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if self.current_frame < EXPLORE_FRAMES:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size[0],self.state_size[1],4))
        update_target = np.zeros((self.batch_size, self.state_size[0],self.state_size[1],4))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            #pick up N continual states as input where N = self.state_stack
            update_input[i] = np.float32(minibatch[i][0])/255.

            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            
            #N continual output states
            update_target[i] = np.float32(minibatch[i][3])/255.

            done.append(minibatch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)


        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            Qvalue = target[i][action[i]]
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))
            

        # and do the model fit!
        hist = self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        self.loss = hist.history['loss']

def preprocess(state):
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.uint8(rgb2gray(state)[35:195:2,0:160:2])


if __name__ == "__main__":

    env = gym.make(GAME)
    # set size of state after preprocess
    state_size = (80,80)
    # get size of action from environment
    action_size = env.action_space.n
    # create a DQN agent
    agent = DQNAgent(state_size, action_size)
    # record scores in every episode
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        for f in range(100000):
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            if f%4 == 0:
                action = agent.get_action()

            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(action, reward, next_state, done)

            # do the training
            
            # record total rewards
            score += reward
            

            if done:
                str = "episode: {}  score: {}  memory length: {} epsilon: {} Q:{} loss: {}".format(
                        e,score,len(agent.memory),agent.epsilon,agent.Q,agent.loss)
                print(str)

                break

# target network
# larger experience replay
# huber loss
# image preprocessing
