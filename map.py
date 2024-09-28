# Importing the libraries
import numpy as np
from random import random, randint, choice
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.metrics import Metrics

# Importing the Dqn object from our AI in ai.py
from ai import *

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', int(1130/1.25))
Config.set('graphics', 'height', int(667/1.25))

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
max_timesteps = 500000
batch_size = 100
discount = 0.99 
tau = 0.005 
policy_noise = 0.2 
noise_clip = 0.5 
policy_freq = 2 
eval_freq = 5e3
file_name = 'TD3_Car_Map_0'
start_timesteps = 15000
expl_noise = 0.1
max_episode_steps = 1000
done = True
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_timesteps = 0
episode_reward = 0
obs = np.array([])
save_models = True

if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

replay_buffer = ReplayBuffer()
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = T3D(6,1,5)
if os.path.isfile('last_actor.pth') and os.path.isfile('last_critic.pth'):
    print(' Attempting to load models before starting ')
    brain.load_models()
else:
    print(' Attempt to load models failed...no checkpoint found..')

last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
initial_start_points = [(195, 110), (460, 460), (995, 520), (195,100), (550,440), (1000,300), (900,330), (500,400), (100,230), (700,400)]
goal_list = [(195, 110), (460, 460), (995, 520), (195,100), (550,440), (1000,300), (900,330), (500,400), (100,230), (700,400)]

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_cord = goal_list[0]
    goal_x, goal_y = goal_cord
    first_update = False
    global location_index
    location_index = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Goal(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.center = initial_start_points[0]
        self.car.velocity = Vector(6, 0)

    def reset(self, is_initial_steps):
        '''
        Reseting the environment and agent and initalising the state
        @param self: imagepatch, orientation, change in distance
        @return: state
        '''
        global goal_x
        global goal_y
        global last_distance
        global location_index  

        if is_initial_steps:
            self.car.center = initial_start_points[location_index]
            location_index += 1 
            if location_index == 9:
                location_index = 0  
        else:
            self.car.center = choice(initial_start_points)
            
        xx = goal_x - self.car.x 
        yy = goal_y - self.car.y         
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        delta = last_distance - distance    
        state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation, delta]
        return state 
    
    def step(self, action):

        global goal_x
        global goal_y
        global last_distance
        global done

        rotation = action.item()
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        self.goal.pos = (goal_x, goal_y)

        xx = goal_x - self.car.x 
        yy = goal_y - self.car.y         
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        delta = last_distance - distance    
        state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation, delta]

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            #print(1, sand[int(self.car.x),int(self.car.y)])
            last_reward = -0.8
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward =  -0.1
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            #print(0, sand[int(self.car.x),int(self.car.y)])
            if distance < last_distance:
                last_reward = 0.8

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -0.5
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -0.5
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -0.5
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -0.5

        if distance < 35:
            print(f'Goal achieved: {goal_x},{goal_y}')
            goal_cord = choice(goal_list)
            goal_x, goal_y = goal_cord
            last_reward = 10
        
        if episode_timesteps + 1 == max_episode_steps:
            done = True

        last_distance = distance

        return state, last_reward, done

    def evaluate_policy(self, brain, eval_episodes=10):
        '''
        calculting average reward for few episodes
        @param brain: TD3 action selection
        @param eval_episodes: 
        @return: avg_reward 
        '''
        avg_reward = 0.
        for _ in range(eval_episodes):
            observation = self.reset(False) # ToDo reset env
            done = False
            while not done:
                action = brain.select_action(np.array(observation))
                observation,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global done
        global max_timesteps
        global replay_buffer
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global episode_timesteps
        global episode_reward
        global obs
        global evaluations

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            evaluations = [self.evaluate_policy(brain)]

        if total_timesteps < max_timesteps:

            # If the episode is done
            if done:

                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(brain))
                    brain.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                if total_timesteps < start_timesteps:
                    obs = self.reset(True)
                else:
                    obs = self.reset(False)

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before start timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-5, high=5, size=(1,))
            else: # After start timesteps, we switch to the model
                action = brain.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(-5, 5)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done = self.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
        
        else:
            action = brain.select_action(np.array(obs))            
            new_obs, reward, done = self.step(action)
            obs = new_obs
            total_timesteps += 1
            if total_timesteps%1000==1:
                print(total_timesteps)

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()