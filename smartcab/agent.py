import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.trial_num = 0 # trial_num


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        #print("updating epsilon")
        #print("testing is ", testing)
        
        if testing==True:
            self.epsilon=0
            self.alpha=0
            return None
        
        self.trial_num+=1
        
        decay = 'cosine' # possible decay functions. Can choose from 'geometric', 'linear', 'cosine', 'tanh','no_decay', 'step', 'default'
        
        if decay == 'geometric':
            kappa = 0.9994 # geometric decay constant
            self.epsilon*=kappa
            #self.alpha*=kappa
        elif decay == 'linear':
            kappa=0.000192  # linear deacy constant for epsilon
            self.epsilon= 1- kappa*self.trial_num
        elif decay == 'cosine':
            kappa=0.0003
            #sigma=0.99
            #self.epsilon-=kappa*(1-sigma* self.epsilon**2)**(0.5)
            self.epsilon = np.cos(kappa*self.trial_num)
        elif decay == 'tanh':
            kappa = 5000000000
            #self.epsilon-= 3*(kappa**(-0.25))*((np.arctanh(self.epsilon-0.01))**1.5)*(1-0.99*self.epsilon**2)
            self.epsilon=np.tanh(kappa/(self.trial_num**3))
        elif decay == 'no_decay':
            if self.trial_num > 5000:
                self.epsilon=0.004
        elif decay == 'step':
            if self.trial_num > 300 and self.trial_num <400:
                self.epsilon = 0.25
            elif self.trial_num >=400 and self.trial_num < 500:
                self.epsilon = 0.042
            elif self.trial_num >= 500:   
                self.epsilon = 0.004
        elif decay == 'default':
            kappa=0.05
            self.epsilon-=kappa
        
                
       
        # If at somepoint epsilon becomes negative due to the linear decadance, then we just set it to zero
        if self.epsilon < 0:
            self.epsilon = 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        #state = (waypoint, inputs['light'], inputs['left'],inputs['right'], inputs['oncoming'], deadline)
        #state = (waypoint, inputs['light'],inputs['left'], inputs['oncoming'], deadline)
        #state = (waypoint, inputs['light'],  inputs['oncoming'])
        state = (waypoint, inputs['light'],  inputs['oncoming'], inputs['left'])

        return state


    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        
        maxQ = max(self.Q[state].values()) 

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        #print("self.learning", self.learning)
        #input("Press Enter")
        
        #print("length of self.Q:", len(self.Q))
        
        if self.learning==False:
            return
        
        if (state in self.Q)==False:
            #print("CREATING a new state entry")
            action_Qval=dict.fromkeys(self.valid_actions, 0.0) #dictionary to store the Q-value for each action
            self.Q[state]=action_Qval  
        
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
                
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        if self.learning == False:
            action = random.choice(self.valid_actions)
            #action = None
        else:
            #print("epsilon:", self.epsilon)
            explore = 1-np.random.choice(a=[0,1], 
                                         size=1, p=[self.epsilon, 1-self.epsilon])[0]
            if explore==1:
                action=random.choice(self.valid_actions)
            else: 
                qmax=self.get_maxQ(state)
                actions_maxQ=[action_value[0] for action_value in filter(lambda x: x[1]==qmax, self.Q[state].items())]
                action=random.choice(actions_maxQ)

        
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        # Implementing the Temporal Difference Learning algorithm i.e. eq. 16-5 in Aurelien Geron's book
        # we will set gamma = 0, since we have been instructed to not use the discount factor 
        
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*reward

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    num_dummies=100
    env = Environment(num_dummies=num_dummies)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    learning = True
    alpha = 0.5
    agent = env.create_agent(LearningAgent, learning=learning, alpha = alpha)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    enforce_deadline=True
    env.set_primary_agent(agent, enforce_deadline)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    display=False
    update_delay=0.00001
    log_metrics=True
    optimized=True
    sim = Simulator(env, update_delay=update_delay, display=display, log_metrics=log_metrics, optimized=optimized)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    n_test=1000
    tolerance = 0.04
    sim.run(n_test=n_test, tolerance=tolerance)

# check out  https://stackoverflow.com/questions/419163/what-does-if-name-main-do to understand the significance of __name__=='__main__' in the following piece of code 

if __name__ == '__main__':
    run()
