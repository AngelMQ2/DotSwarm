from tkinter import SEL
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset, DataSet
import time
from queue import PriorityQueue

# Environmen hyperparam:
#ARENA_SIDE_LENGTH = 100
#BLOCK_SIZE        = 5
WALLS_ON          = True
STEPS             = 1000

# Agent hyperparameters:
MAX_SPEED         = 5e-2
LIDAR_RANGE       = 5
NUM_RAYS          = 8
SAFE_SPACE        = 2       # Self safe-space, avoid collitions



# Make the environment toroidal 
def wrap(z, arena_side_lenghth):    
    return z % arena_side_lenghth


class agent:
    def __init__(self, id, x, y, vx, vy, home, map, DATASET, num_agent):

        # Operation parameter 
        self.id = id         # Agent ID   
        self.mode = "stop"   # Operation mode (behaviour)d
        self.N = None        # List of neighbors

        ################### AGENT MOTION #######################
        # Initial position:
        self.x = x
        self.y = y
    
        # Velocity:
        self.vx = vx
        self.vy = vy
        self.V = np.linalg.norm([self.vx,self.vy])    # Forward velocity
        self.yaw = np.arctan2(self.vy,self.vx)        # Agent orientation

        self.vx0 = vx   # Random fixed value
        self.vy0 = vy   # Random fixed value

        # Low-pass filter:
        self.last_vx = self.vx
        self.last_vy = self.vy
        
        ################### LIDAR DATA STRUCTURE ##################
        # LIDAR datastructure:
        self.scan = np.zeros((NUM_RAYS,LIDAR_RANGE))
        self.obstacle_score = 0     # Counter to determine wheter there is a infront obstacle

        ##################### DATASET #########################

        # World simulation: BW map to measure obstacles, DATASET to read temperature values
        self.map = map.map            # Black and white image used to navigate
        self.world = DATASET          # External world information (simulate monitoring process)
        self.num_agent = num_agent
        self.time = time.time()

        # Agent individual dataset:
        self.dataset = DataSet(map,blind_copy=True)
        self.data_to_save = 0                                       # Count number of new data --> used to come back home
        self.update_plot = False                                    # Bool to plot the dataset
        

        # Home position:
        self.home = self.dataset.get_centroid(home[0],home[1]) # Centroid 
        self.at_home = False

        # Variables for exploration
        self.previous_location = self.dataset.get_centroid(self.x,self.y)
        self.desired_point = None

        # Variables for A-Star
        self.path_to_home = []

    # Return current position
    def position(self):
        return np.array([self.x, self.y])
    
    # Set operation mode to follow
    def set_mode(self, mode):
        self.mode = mode

     # Return list of neighbors 
    def neightbors(self):
        return self.N
    
    # Set list of neighbors
    def set_neightbors(self,neightbors):
        self.N = neightbors

    # Function to normalize and scale the command of velocity accordin to the MAX_SPEED
    def norm_velocity(self):
        if self.vx != 0 and self.vy != 0:
            # Normalize and scale:
            self.vx = MAX_SPEED*(self.vx/np.linalg.norm([self.vx,self.vy]))
            self.vy = MAX_SPEED*(self.vy/np.linalg.norm([self.vx,self.vy]))

        self.V = np.linalg.norm([self.vx,self.vy])    # Forward velocity
        self.yaw = np.arctan2(self.vy,self.vx)        # Agent orientation

    def low_pass_filter(self):
        w = 0.1    # Weighted new velocity

        # Filtered velocity:
        new_vx = self.vx*w + self.last_vx*(1-w)
        new_vy = self.vy*w + self.last_vy*(1-w)

        # Store values
        self.vx = new_vx
        self.vy = new_vy
        self.last_vx = new_vx
        self.last_vy = new_vy

    ################### MAIN OPERATION FUNCTION ########################

    def one_step(self):
        # 1º Read LIDAR:
        self.lidar_scan()

        # 2º Take action based on current behavior:
        if self.mode == "random":
            self.forward()

        elif self.mode == "cluster":
            self.cluster()

        elif self.mode == "home":
            self.homing()

        elif self.mode == "dispersion":
            self.dispersion()

        elif self.mode == "explore":
            if self.data_to_save > 5:   # If number of data discover greater than 5 come back home for updating map
                self.homing()
            else:
                self.exploration()

        else:   # Do nothing
            self.stop()

        if self.mode != "explore": self.desired_point = None    # Needed to reset exploration

        # 3º Check collition:
        self.avoid_collision()     

        # 4º Normalize velocity vector:
        self.norm_velocity()
        self.low_pass_filter()  #Low-pass filter - smooth movements

        # 5º Update state:
        self.x = wrap(self.x + self.vx, self.dataset.arena_size)
        self.y = wrap(self.y + self.vy, self.dataset.arena_size)
        self.V = np.linalg.norm([self.vx,self.vy])    # Forward velocity
        self.yaw = np.arctan2(self.vy,self.vx)        # Agent orientation

        # 6º Check if close to home position:
        if np.linalg.norm([self.x-self.home[0],self.y-self.home[1]]) < self.dataset.K:  # If smaller than block size
            self.at_home = True
            self.data_to_save = 0
        else:
            self.at_home = False

        # Check map values:
        self.monitoring()
        self.update_map()
        self.track_location()        

######################## SENSING FUNCTIONS #################################

    # Function to read the world value for the current position
    def monitoring(self):
        if self.dataset.get_value(self.x, self.y) != self.world.get_value(self.x,self.y):   # If not registered/updated
            self.update_plot = True     # Enable plot
            self.data_to_save += 1      # New data discovered
            value = self.world.get_value(self.x, self.y)                    # Read value from world dataset
            self.dataset.store_value(self.x, self.y, value, True, True)     # Save value in individual dataset

    # Function to simulate the LIDAR reading cycle
    # NOTE: Lidar fixed orientation, not rely on agent's yaw angle
    def lidar_scan(self):
        # Sweep circumference centred on the agent 
        for i,angle in enumerate(np.linspace(0,  2*np.pi, NUM_RAYS, endpoint=False)):
            # Director vector:
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Scan:
            ray = []
            for step in range(LIDAR_RANGE):
                # Compute ray trajectory points:
                x = int(self.x + dx * step)
                y = int(self.y + dy * step)

                if 0 <= x < self.dataset.arena_size and 0 <= y < self.dataset.arena_size:
                    ray.append(self.map[y, x])  # Save Black or white map value for (x,y) in ray trajectory

                    # If (x,y) is a wall then mark block as unreacheable:
                    if self.map[y,x] == 0 and self.dataset.get_reacheable(x,y):
                        self.dataset.store_value(x,y, None, False, False)
                        self.update_plot = True
                else:
                    ray.append(0)  # If ray goes out of bounds, consider it as hitting an obstacle

            # Store ray
            ray = np.array(ray)
            self.scan[i] = ray

    # Update self dataset with neightbors'
    def update_map(self):
        t = time.time()
        if (int(t) + self.id) % self.num_agent == 0 and time.time() - self.time > 5:     # If 50 sec from prev. update
            #print(f"Updating map agent {self.id} time {t} ")
            self.time = t
            for n in self.N:
                self.dataset.merge(n.dataset)   # Merge self map and neightbor

    ################### BEHAVIOR FUNCTIONS #####################

    def avoid_collision(self):
        self.norm_velocity()    # Need to normalize vector
       
        if self.vx != 0 and self.vy != 0:
            # AVOID COLLISION WITH NEIGHBORS
            for n in self.N:
                distance = np.linalg.norm(self.position() - n.position()) + 0.001
                if distance < SAFE_SPACE:
                    # Calculate the adjustment vector
                    adjustment = [float(diff) for diff in (self.position() - n.position())]
                    adjustment /= distance  # Normalize to unit vector
                    adjustment *= self.V*np.exp((SAFE_SPACE-distance)/(distance))   # Scale based on exponential inversely proportional to distance
                    self.vx += adjustment[0]
                    self.vy += adjustment[1]
            
            # AVOID COLLISION WITH ENVIRONMENT OBSTACLES
            adjustment = [0,0]
            for i,angle in enumerate(np.linspace(0, 2*np.pi, NUM_RAYS, endpoint=False)):
                # Director vector:
                dx = -np.cos(angle)
                dy = -np.sin(angle)
                # Count non-cero element in the scan - correspond to distance to distance
                distance = np.count_nonzero(self.scan[i])
                if distance < LIDAR_RANGE:
                    scale = self.V*np.clip(np.exp(1/np.max([0.1,distance])),0,3)/2   # Scale based on exponential inversely proportional to distance
                    self.vx += scale*dx
                    self.vy += scale*dy

    # Behavior to move randomly. If agent is stoped then asign random stored velocity
    # NOTE: Emerging behavior such that the avoid collision function modify the velocity vector freely. Result is agent moving forward and bouncing in obstacles
    def forward(self):
        # Randomly change the velocity:
        if self.vx == 0 and self.vy == 0:
            self.vx = self.vx0
            self.vy = self.vy0

    # Function to stop the agent
    def stop(self):
        self.vx = 0
        self.vy = 0

    # Behavior to come back home
    def homing(self):
     
        if self.path_to_home == [] and not self.at_home:
            self.path_to_home = self.a_star(self.home)
            #print("A*")
            #print(f"\n{self.id}, x,y {[self.x,self.y]}, path {self.path_to_home}")

        if self.at_home:
            if self.path_to_home != []:
                self.path_to_home = [] 

        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0
        [x,y] = self.position()


        if self.path_to_home != []: # Check if self.path_to_home is empty

            next_centroid = self.path_to_home[0]

            if np.linalg.norm([self.x-next_centroid[0],self.y-next_centroid[1]]) < 3 and self.path_to_home != []:  # If smaller than block size
                self.path_to_home.pop(0)
                next_centroid = self.path_to_home[0]
        
            if not self.dataset.get_reacheable(next_centroid[0], next_centroid[1]):
                self.path_to_home = self.a_star(self.home)
        
            # Distance to desired point:     
            delta_x = (next_centroid[0] - x)
            delta_y = (next_centroid[1] - y)

        else:
            delta_x = (self.home[0] - x)
            delta_y = (self.home[1] - y)

        # Modify velocity vector
        self.vx = delta_x*MAX_SPEED
        self.vy = delta_y*MAX_SPEED

    # Function to gather neighbor agents into a region
    def cluster(self):
        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0
        [x,y] = self.position()
        
        if self.neightbors() == []: # If no neightbor move fordward freely
            self.forward()
        else:
            # Control law:
            for n in self.neightbors():
                [n_x, n_y] = [n.x,n.y]
                delta_x += (n_x - x)
                delta_y += (n_y - y) 
            self.vx = delta_x
            self.vy = delta_y
    
    def dispersion(self):
        # Difference position neightbors-agent
        umbral = 10

        repulsion_factor = 0.2
        atraction_factor = -0.005
        
        # Deal with too close agents:
        neightbors_too_close = 0
        close_point = (0,0)

        # Control law:
        dispersion_force=(0,0)
        for n in self.neightbors():
            diff =  [float(diff) for diff in (self.position() - n.position())]  # Difference vector
            dist = np.linalg.norm(diff) # Norm
            diff /= dist                # Normalize difference vector

            # Determine scale and direction (repulsion or atraction)
            if dist < umbral:   # Repulsion
                if dist > SAFE_SPACE:   # Not too close neightbor
                    dispersion_force += diff*repulsion_factor
                else:                   # In case 2 agent too close together then first move away before having others into account
                    neightbors_too_close += 1
                    close_point = diff*repulsion_factor        
            else:   # Atraction
                dispersion_force += diff*atraction_factor

        # Asign velocity:
        if neightbors_too_close != 1:
            self.vx = dispersion_force[0]
            self.vy = dispersion_force[1]
        else:
            self.vx = close_point[0]
            self.vy = close_point[1]


    def track_location(self):
        # Undo the forward step:
        prev_x = self.x - self.vx
        prev_y = self.y - self.vy
        prev_index = self.dataset.index(prev_x,prev_y)
        current_index = self.dataset.index(self.x, self.y)

        if prev_index != current_index:
            self.previous_location = self.dataset.get_centroid(prev_x, prev_y)
    
    # NOTE: Choose blocks randomly from a list of (Reacheable,Non-explored) for exploration algorithm
    # Filter first those block that are reacheables, then the non-explored (else reacheables only). List of candidates block, choose randomly weighted
    def choose_next_block(self):
        #self_previous_idx = self.dataset.index(self.previous_location[0],self.previous_location[1])
        #nodes_to_explore = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        
        self_idx = self.dataset.index(self.x, self.y)           # Get current block indexs
        nodes_to_explore = [(1, 0), (0, 1), (0, -1), (-1, 0)]   # 4-neightbor to explore

        # Nodes candidates:
        reacheable_centroids = []
        non_visited_centroids = []  

        for node_idx in nodes_to_explore:
            # Index of nodes to explore
            i = self_idx[0] + node_idx[0]
            j = self_idx[1] + node_idx[1]

            # Check boundaries:
            size = self.dataset.size()
            if i < size[0] and j < size[1]: #and i != self_previous_idx[0] and j != self_previous_idx[1]:
                info = self.dataset.get_info(i,j)
                centroid = (info.centroid[1], info.centroid[0])
                # Keep only reacheable:
                if info.reacheable:
                    # Keep only un-discovered:
                    reacheable_centroids.append(centroid)
                    if not info.visited:
                        non_visited_centroids.append(centroid)
 
        if non_visited_centroids == []:    # If no un-visited block around just go forward freely
            self.desired_point = None
            return False
        
        else:                              # If any un-visited block around
            #Distance to candidate centroids
            dist_array = np.array([np.linalg.norm([self.x - centroid[0], self.y - centroid[1]]) for centroid in non_visited_centroids])
            # Prior probability - greater probability those centroid further away from current position
            prior_prob = np.array(dist_array/np.sum(dist_array))
            selected_idx = np.random.choice(range(len(non_visited_centroids)), p = prior_prob)
            self.desired_point = non_visited_centroids[selected_idx]
            
            return True
       
    # Behavior of monitoring
    def exploration(self):
        found_point = False
        # Check if desired point has not been set / current desired point is not reacheable
        if self.desired_point == None or not self.dataset.get_reacheable(self.desired_point[0],self.desired_point[1]):
            found_point = self.choose_next_block()

        # If not unvisited block to explore just go forward
        if found_point == False:
            self.forward()
        else:
            # Control law:
            [x,y] = self.position()
            delta_x = self.desired_point[0] - x
            delta_y = self.desired_point[1] - y

            # Check if has reach point:
            if np.linalg.norm([self.desired_point[0]-x, self.desired_point[1]-y]) < SAFE_SPACE:
                self.desired_point = None # Choose new point to tracj

            self.vx = delta_x
            self.vy = delta_y

    ##################### A* Algorithm ###############################################

    # Calculate the manhatten norm for point a and b
    def manhatten_norm(self, point_a, point_b):
        x_a, y_a = point_a
        x_b, y_b = point_b

        return abs(x_a - x_b) + (y_a - y_b)

    # returns a list of points from start to finish
    def a_star(self, goal):
        # start point is the point of the agent right now
        g_score = {}
        f_score = {}
        start = self.dataset.get_centroid(self.x, self.y)
        for i in range(self.dataset.dim):
            for j in range(self.dataset.dim):
                info = self.dataset.get_info(i, j)
                if info.reacheable:
                    g_score[(info.centroid[1], info.centroid[0])] = float("inf") # Generate Dictonary where centriod is the key and the value is the A* g value
                    f_score[(info.centroid[1], info.centroid[0])] = float("inf") # Generate Dictonary where centriod is the key and the value is the A* f value

        g_score[start] = 0
        f_score[start] = self.manhatten_norm(start, goal)
        reverse_path = {}

        pqueue = PriorityQueue()
        # Save values: f_score      h_score                                 centriod of point
        pqueue.put((f_score[start], self.manhatten_norm(start, goal), start)) 
        while not pqueue.empty():
            curr_centroid = pqueue.get()[2] # Get the centriod of the current cell
            if curr_centroid == goal: # End when at home with current cell
                break

            self_idx = self.dataset.index(curr_centroid[0], curr_centroid[1])
            nodes_to_explore = [(1, 0), (0, 1), (0, -1), (-1, 0)]

            
            for node_idx in nodes_to_explore:
            # Index of nodes to explore
                i = self_idx[0] + node_idx[0]
                j = self_idx[1] + node_idx[1]

                # Check to be inside the arena
                size = self.dataset.size()
                if i < size[0] and j < size[1]:

                    info = self.dataset.get_info(i,j)
                    if info.reacheable:
                        next_centroid = (info.centroid[1], info.centroid[0])

                        temp_g_score = g_score[curr_centroid] + 4 # TODO: Maybe +1 or something
                        temp_f_score = temp_g_score + self.manhatten_norm(next_centroid, goal)


                        if temp_f_score < f_score[next_centroid]:
                            g_score[next_centroid] = temp_g_score
                            f_score[next_centroid] = temp_f_score
                            pqueue.put((temp_f_score, self.manhatten_norm(next_centroid, goal), next_centroid))
                            reverse_path[next_centroid] = curr_centroid
                        

        # Path is currently saved in reversed, make it forward
        fwd_path_array = []
        temp_centroid = goal
        while temp_centroid != start:
            fwd_path_array.append(temp_centroid)
            temp_centroid = reverse_path[temp_centroid]

        fwd_path_array.append(start)
        fwd_path_array.reverse()

        return fwd_path_array

class SwarmNetwork():
    def __init__(self, home, map, dataset, num_agent, communication_radium, start_home = False):

        # Save hyperparameters:
        self.num_agents = num_agent
        self.neightbor_space = communication_radium

        #################### CREATE AGENTS #####################################
        # Set random intial point:
        x_0 = [] 
        y_0 = [] 

        # Ensure not to spawn in wall
        self.map = map.map
        for i in range(self.num_agents):
            if start_home == True:
                x, y = home
            else:
                x, y = self.generate_randome_start(dataset.arena_size)
            x_0.append(x)
            y_0.append(y)

        # list of random velocities:
        vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(self.num_agents,))
        vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(self.num_agents,))

        # Create agents::
        self.index = np.arange(self.num_agents)
        self.agents = [agent(i,x_0[i],y_0[i],vx[i],vy[i],home,map,dataset, num_agent) for i in range(self.num_agents)] # List of agents (own self-position)

        # Adjacency matrix - used for evaluation
        self.Adj = np.zeros((self.num_agents,self.num_agents))  

        # Set initial topology - neighboors relationships
        self.update_Topology()

        ################# DATASET ##################
        self.global_map = DataSet(map,blind_copy=True, save_home= True)
        self.update_map()
        self.init_timer = time.time()
        self.update_timer = time.time()
        
    
    def generate_randome_start(self, arena_size):
        x = np.random.randint(low=0, high=arena_size)
        y = np.random.randint(low=0, high=arena_size)

        if self.map[y,x] == 0:
            return self.generate_randome_start(arena_size)
        else:    
            return x, y
        
    # Return list of agents position
    def state(self):
        return np.array([agent.position() for agent in self.agents])

    ########## MAIN FUNCTION ###################
    def one_step(self, mode = "stop"):
        t = time.time()
        for agent in self.agents:
            agent.set_mode(mode)    # Control behavior
            agent.one_step()        # Agent main function
            
            if agent.at_home == True and abs(t - self.update_timer) > 1.5:   # If agent at home call update ground station map
                #show = abs(t - self.init_timer) > 10
                # Update overall map:
                self.global_map.merge(agent.dataset, show = False)
                print(self.global_map)
                self.update_timer = time.time()

        # Update agents neightbourhood:
        self.update_Topology()

    # First update of every map:
    def update_map(self):
        for agent in self.agents:
            self.global_map.merge(agent.dataset,show=False)

    # Update neighbors relationship        
    def update_Topology(self):
        self.Adj = np.eye(self.num_agents*self.num_agents)

        state = self.state()
        neightbors = [] # List of list of neightbors

        # For every agent in the swarm
        for agent in self.agents:
            # Check distance to every other agent
            dist_neighbors = np.linalg.norm(agent.position() - state,axis=1)
            # Select closest agents:
            sort_id = self.index[np.argsort(dist_neighbors)[1:]]
            neightbors_id = []
            for idx in sort_id:
                if dist_neighbors[idx] < self.neightbor_space:
                    neightbors_id.append(idx)

            neightbors.append(neightbors_id)

        # Save list of agents as every agent neightbors:
        for i,agent in enumerate(self.agents):
            temp_neightbor = []
            for other_agent in self.agents:
                if other_agent.id in neightbors[i]: 
                    temp_neightbor.append(other_agent)
                    self.Adj[agent.id, other_agent.id] = 1
                    self.Adj[other_agent.id, agent.id] = 1 
            # Update agent's neightbour:
            agent.set_neightbors(temp_neightbor)
            