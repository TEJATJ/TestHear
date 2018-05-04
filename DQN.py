import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
EPISODES = 1000
import traceback

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = 2
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # self.graph = tf.get_default_graph()
        # with K.get_session().graph.as_default() as g:
        self.model = self._build_model()
        print(self.model.summary())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Conv2D(16,kernel_size=(15,15),input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(10,10)))
        model.add(Conv2D(10,kernel_size=(5,5)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        with K.get_session().graph.as_default() as g:
            return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # print("helolu")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with K.get_session().graph.as_default() as g:
            try:
                act_values = self.model.predict(state)
            except Exception as e:
                print(e)
                os._exit(1)


            print(act_values)
            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        with K.get_session().graph.as_default() as g:
            for state, action, reward, next_state, done in minibatch:
                if(state is None):
                    continue
                target = reward
                if not done:
                    print(next_state.shape)
                    try:
                        target = (reward + self.gamma * np.amax(self.model.predict(next_state)))
                    except Exception as e:
                        print("1-{}".format(e))
                        os._exit(1)

                try:

                    target_f = self.model.predict(state)
                except Exception as e:
                    print(state)
                    print("2-{}".format(e))
                    os._exit(1)
                # print("target{}-action-{}".format(target,action))
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=1)

        print("model fitted")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
