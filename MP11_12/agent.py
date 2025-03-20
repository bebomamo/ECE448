import numpy as np
import utils
from utils import UP, DOWN, LEFT, RIGHT


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table.
        self.N[state + (action,)] += 1
        return

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table.
        # learning_rate = C / (C + N(s,a))
        learning_rate = self.C / (self.C + self.N[s+(a,)])
        max = float('-inf')
        directions = [UP, DOWN, LEFT, RIGHT]
        for direction in directions:
            if self.Q[s_prime + (direction,)] > max:
                max = self.Q[s_prime + (direction,)]
        self.Q[s+(a,)] = self.Q[s+(a,)] + learning_rate*(r + self.gamma*max - self.Q[s+(a,)])
        return      

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment
        # state format: (FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X, ADJOINING_WALL_Y, 
        #                  ADJOINING_BODY_TOP, ADJOINING_BODY_BOTTOM, 
        #                   ADJOINING_BODY_LEFT,ADJOINING_BODY_RIGHT,ACTIONS)
        FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X, ADJOINING_WALL_Y, ADJOINING_BODY_TOP, ADJOINING_BODY_BOTTOM, ADJOINING_BODY_LEFT,ADJOINING_BODY_RIGHT = 0,1,2,3,4,5,6,7
        state = [0,0,0,0,0,0,0,0]
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment
        # food dirs
        if(snake_head_x > food_x):
            state[FOOD_DIR_X] = 1
        elif(snake_head_x < food_x):
            state[FOOD_DIR_X] = 2
        if(snake_head_y > food_y):
            state[FOOD_DIR_Y] = 1
        elif(snake_head_y < food_y):
            state[FOOD_DIR_Y] = 2

        # adjoining walls
        rock_x_list = [rock_x, rock_x+1]
        # THE X PART
        if snake_head_y == rock_y: 
            for cur_rock_x in rock_x_list:
                if snake_head_x == cur_rock_x+1:
                    state[ADJOINING_WALL_X] = 1
            if state[ADJOINING_WALL_X] == 0:
                if snake_head_x == 1:
                    state[ADJOINING_WALL_X] = 1
                elif snake_head_x == self.display_width - 2:
                    state[ADJOINING_WALL_X] = 2
                else:
                    for cur_rock_x in rock_x_list:
                        if snake_head_x == cur_rock_x-1:
                            state[ADJOINING_WALL_X] = 2
        else:
            if snake_head_x == 1:
                state[ADJOINING_WALL_X] = 1
            elif snake_head_x == self.display_width - 2:
                state[ADJOINING_WALL_X] = 2  
        # THE Y PART
        for cur_rock_x in rock_x_list: 
            if snake_head_x == cur_rock_x:
                if snake_head_y == rock_y+1:
                    state[ADJOINING_WALL_Y] = 1
                elif snake_head_y == 1:
                    state[ADJOINING_WALL_Y] = 1
                elif snake_head_y == self.display_height - 2:
                    state[ADJOINING_WALL_Y] = 2
                elif snake_head_y == rock_y-1:
                    state[ADJOINING_WALL_Y] = 2
        else:
            if snake_head_y == 1:
                state[ADJOINING_WALL_Y] = 1
            elif snake_head_y == self.display_height - 2:
                state[ADJOINING_WALL_Y] = 2  

        # adjoining body
        up, down, left, right = (snake_head_x, snake_head_y-1), (snake_head_x, snake_head_y+1), (snake_head_x-1, snake_head_y), (snake_head_x+1, snake_head_y)
        for position in snake_body:
            if position == up:
                state[ADJOINING_BODY_TOP] = 1
            if position == down:
                state[ADJOINING_BODY_BOTTOM] = 1
            if position == left:
                state[ADJOINING_BODY_LEFT] = 1
            if position == right:
                state[ADJOINING_BODY_RIGHT] = 1

        return tuple(state)
