from PIL import ImageGrab,Image,ImageFilter
import numpy as np
from pynput.keyboard import Key,Controller
import re
from DQN import DQNAgent as D
import tensorflow as tf
from keras import backend as K
keyboard=Controller()
from queue import Queue
import cv2
import os
from threading import Thread
from mss import mss
from pytesseract import image_to_string
from datetime import datetime
import time
EPISODES=50
# tools = pyocr.get_available_tools()[0]
# builder=pyocr.builders.DigitBuilder()
sct = mss()
state_size=(300,450,1)
monitor={'top': 90, 'left': 0, 'width': state_size[1], 'height': state_size[0]}
q=Queue(maxsize=0)
train_q=Queue(maxsize=0)
class Start():
    def __init__(self):
        self.start=False


start=Start()
def takeAction(action=0):
    if(action==1):
        keyboard.press(Key.space)
        keyboard.release(Key.space)
    else:
        None
agent=D(state_size=state_size,action_size=2)
agent.batch_size=10
def processQueue(thread_name,queue,train_q,start,agent):
    previous=None
    previous_action=None
    previous_reward=0
    start_state=True
    i=0
    done=False
    while(1):
        try:
            if(start.start):
                i+=1

            q=queue.get()

            score=image_to_string(q['image'],lang="eng")
            action=q['action']
            pattern=re.findall(r"[0-9][0-9][0-9][0-9][0-9]",score)

            reward=0
            if(len(pattern)==1):
                reward=int(pattern[0])
            elif(len(pattern)==2):
                reward=int(pattern[1])
            next_state=None

            if(reward==previous_reward):
                done=True
            else:
                done=False
            # print(reward,done)
            start_state=True
            state=np.array(q['previous_image'].convert("L")).reshape((1,state_size[0],state_size[1],1))
            next_state=np.array(q['image'].convert("L")).reshape((1,state_size[0],state_size[1],1))
            reward_k=reward if not done else -100
            # if(not start_state):
            agent.remember(state,action,reward_k*10,next_state,done)
            start_state=False
            previous=next_state
            previous_action=action
            previous_reward=reward
            queue.task_done()



            if(i==100):
                print("Queue Reached 500")
                start.start=False
                i=0
        except Exception as e:
            print(e)
            os._exit(1)


        # print(queue.qsize())

        # print(queue.qsize())

def showImage(thread,queue,train_q,start,agent):
    last_time=datetime.now()
    i=0
    batch_size=50

    # graph=K.get_session().graph
    # agent.graph=graph
    while 1:
        # img = ImageGrab.grab(bbox=(0,80,370,270)) #bbox specifies specific region (bbox= x,y,width,height)

        # print(type(image_pil))

        agent.state_size=state_size
        # score=image_to_string(img,lang="eng",)
        # pattern=re.findall(r"[0-9][0-9][0-9][0-9][0-9]",score)
        # game_over=re.findall(r"g|G",score)
        # reward=0
        # if(len(pattern)>1):
        #     reward=int(pattern[1])
        # if(len(game_over)!=0):
        #     keyboard.press(Key.space)
        #     keyboard.release(Key.space)
        # frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2GRAY)

        if(start.start==True):
            img=sct.grab(monitor)

            img = Image.frombytes('RGB', (img.width, img.height), img.rgb)
            image_pil=np.asarray(img.convert('L'))
            image=image_pil.reshape((1,state_size[0],state_size[1],1))
            with tf.get_default_graph().as_default():
                action=agent.act(image)
            takeAction(action)
            time.sleep(0.3)
            next_img=sct.grab(monitor)
            next_img = Image.frombytes('RGB', (next_img.width, next_img.height), next_img.rgb)
            image_pil_next=np.asarray(next_img.convert('L'))
            queue.put({"previous_image":img,"image":next_img,"action":action})
            cv2.imshow("test",image_pil_next )
        elif(start.start==False):
            print("Training Episode {}".format(i))
            for e in range(EPISODES):
                if(len(agent.memory)>batch_size):
                    with tf.get_default_graph().as_default():
                        agent.replay(batch_size)
            print("Training Completed...Replaying The Game")
            agent.save('model1.hdf5')
            start.start=True
            # time.sleep(3)



        i+=1
        # time.sleep(0.1)
        # delta=datetime.now()-last_time
        # delta=delta.seconds+ (float(1) / delta.microseconds)
        # print("Loop Took {} seconds".format(delta))
        last_time=datetime.now()
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
worker=Thread(target=processQueue,args=("queue thread",q,train_q,start,agent))
worker.setDaemon(True)
worker.start()
worker1=Thread(target=showImage,args=("queue thread",q,train_q,start,agent))
worker1.setDaemon(True)
worker1.start()
q.join()

worker.join()
worker1.join()
