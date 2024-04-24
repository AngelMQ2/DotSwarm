from re import M
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset, DataSet

# Environmen hyperparam:
ARENA_SIDE_LENGTH = 100
BLOCK_SIZE        = 5
STEPS             = 1000

# Agent hyperparameters:
MAX_SPEED         = 5e-2

BASE_SPEED        = 5e-3
LIDAR_RANGE       = 5
NUM_RAYS          = 8
SAFE_SPACE        = 2            # Self safe-space, avoid collitions

# Swarm hyperparameters:
NUMBER_OF_ROBOTS  = 50
NEIGHTBORS_SPACE  = 20      # Radius of communication area



# Make the environment toroidal 
def wrap(z):    
    return z % ARENA_SIDE_LENGTH

class agent:
    def __init__(self, id, x, y, vx, vy, home, dataset):

        # Agent ID:
        self.id = id

        # Operation mode variable:
        self.mode = "stop"

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

        # Home position:
        self.home = home
        self.at_home = False

        # List of neightbors
        self.N = None 


        # LIDAR datastructure:
        self.scan = np.zeros((NUM_RAYS,LIDAR_RANGE))

        # Agent's dataset - based on a reference's parameter
        self.dataset = DataSet(dataset,copy=True)
        self.data_to_save = 0


    def position(self):
        return np.array([self.x, self.y])

    def set_mode(self, mode):
        self.mode = mode

    def one_step(self):

        # Take action:
        if self.mode == "random":
            if self.data_to_save > 5:   # If number of data discover greater than 5 come back home for updating map
                self.homing()
            else:
                self.forward()
        elif self.mode == "cluster":
            self.cluster()
        elif self.mode == "home":
            self.homing()
        elif self.mode == "dispersion":
            self.dispersion()
        else:   # Do nothing
            self.stop()

        # Update state:
        self.V = np.linalg.norm([self.vx,self.vy])    # Forward velocity
        self.yaw = np.arctan2(self.vy,self.vx)        # Agent orientation

        # Read LIDAR:
        self.lidar_scan()

        # Check collition:
        self.avoid_collision()      

        # Saturate velocity
        if self.vx > MAX_SPEED or self.vy > MAX_SPEED:
            self.vx = MAX_SPEED*(self.vx/np.max([self.vx,self.vy]))
            self.vy = MAX_SPEED*(self.vy/np.max([self.vx,self.vy]))

        # Update state:
        self.x = wrap(self.x + self.vx)
        self.y = wrap(self.y + self.vy)
        self.V = np.linalg.norm([self.vx,self.vy])    # Forward velocity
        self.yaw = np.arctan2(self.vy,self.vx)        # Agent orientation

        # Check if close to home position:
        if np.linalg.norm([self.x-self.home[0],self.y-self.home[1]]) < BLOCK_SIZE:
            self.at_home = True
            self.data_to_save = 0
        else:
            self.at_home = False

        # Check map values:
        self.monitoring()

    def neightbors(self):
        return self.N

    def set_neightbors(self,neightbors):
        self.N = neightbors

    def monitoring(self):
        if not self.dataset.get_state(self.x, self.y):
            self.data_to_save += 1
            value = DATASET.get_value(self.x, self.y)
            self.dataset.store_value(self.x, self.y, value)

    def avoid_collision(self):
        #if self.vx != 0 and self.vy != 0:   # If not stop

        # Avoid collision with other agents:
        for n in self.N:
            distance = np.linalg.norm(self.position() - n.position()) + 0.001
            if distance < SAFE_SPACE:
                # Calculate the adjustment vector
                adjustment = [float(diff) for diff in (self.position() - n.position())]
                adjustment /= distance  # Normalize to unit vector
                adjustment *= (SAFE_SPACE - distance) / 2  # Scale by the amount to adjust
                self.vx += adjustment[0]
                self.vy += adjustment[1]
        
        # Avoid collision with env obstables:
        adjustment = [0,0]
        min_dist = LIDAR_RANGE
        for i,angle in enumerate(np.linspace(self.yaw, self.yaw + 2*np.pi, NUM_RAYS, endpoint=False)):
            # Director vector:
            dx = np.cos(angle)
            dy = np.sin(angle)


            # Count number of ones = distance to wall:
            distance = np.count_nonzero(self.scan[i])
            if distance < min_dist: min_dist = distance
            avoid_factor = LIDAR_RANGE - distance  #TODO: Review cero counting rather than incremental counting
            adjustment[0] += -avoid_factor*dx
            adjustment[1] += -avoid_factor*dy
        
        norm = np.linalg.norm(adjustment)
        if norm != 0:
            # TODO: Improve scale factor
            adjustment /= norm  # Normalize to unit vector
            adjustment *= 2*self.V*((LIDAR_RANGE - min_dist) / LIDAR_RANGE)  # Scale by the amount to adjust
            self.vx += adjustment[0] #*scale#*abs(self.vx) # Scaled by the vlocity abs value
            self.vy += adjustment[1] #*scale#*abs(self.vy)


    def lidar_scan(self):

        # Sweep circumference centred on the agent - starting from the agent's orientation
        for i,angle in enumerate(np.linspace(self.yaw, self.yaw + 2*np.pi, NUM_RAYS, endpoint=False)):
            # Director vector:
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Scan:
            ray = []
            for step in range(LIDAR_RANGE):
                x = int(self.x + dx * step)
                y = int(self.y + dy * step)
                if 0 <= x < ARENA_SIDE_LENGTH and 0 <= y < ARENA_SIDE_LENGTH:
                    ray.append(BW_MAP[y, x])
                else:
                    ray.append(0)  # If ray goes out of bounds, consider it as hitting an obstacle

            ray = np.array(ray)
            self.scan[i] = ray


    def forward(self):
        # Randomly change the velocity:
        if np.random.uniform() > 0.99:
            self.vx0 += np.random.uniform(low=-BASE_SPEED, high=BASE_SPEED)
            self.vy0 += np.random.uniform(low=-BASE_SPEED, high=BASE_SPEED)
        self.vx = self.vx0
        self.vy = self.vy0

  
    def stop(self):
        self.vx = 0
        self.vy = 0
    
    def homing(self):
        
        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0

        [x,y] = self.position()
        
        # Distance to desired point:     
        delta_x = (self.home[0] - x)
        delta_y = (self.home[1] - y)

        self.vx = delta_x*BASE_SPEED
        self.vy = delta_y*BASE_SPEED

    def cluster(self):
        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0

        [x,y] = self.position()
        
        # Control law:
        for n in self.neightbors():
            [n_x, n_y] = [n.x,n.y]
            delta_x += (n_x - x)*BASE_SPEED
            delta_y += (n_y - y)*BASE_SPEED
        

    def homing(self, point = (5,5)):
        delta_x = 0
        delta_y = 0

        [x,y] = self.position()
        [p_x, p_y]  = point
        delta_x += p_x - x
        delta_y += p_y - y

        distance_to_n = []
        for n in self.N:
            distance_to_n.append(np.linalg.norm(self.position() - n.position()))

        ##print(distance_to_n)
        '''if max(distance_to_n) < SAFE_SPACwE +0.1:
            # move in circle
            delat_x = -delat_y
            delat_y = delat_x
'''
        
        x_next = wrap(x + BASE_SPEED*delta_x)
        y_next = wrap(y + BASE_SPEED*delta_y)
        
        self.set_position(x_next,y_next)
        return x_next, y_next

        if self.neightbors() == []:
            delta_x += MAX_SPEED #*np.random.uniform(0,1)
            delta_y += MAX_SPEED #*np.random.uniform(0,1)


        self.vx = delta_x
        self.vy = delta_y

    def dispersion(self):
        # Difference position neightbors-agent
        delta_x = 0
        delta_y = 0
        umbral = NEIGHTBORS_SPACE/2

        [x,y] = self.position()

        dist=0
        
        # Control law:
        for n in self.neightbors():
            [n_x, n_y] = [n.x,n.y]
            dist=np.linalg.norm(self.position()-[n_x, n_y])
            delta_x += (n_x - x)/dist
            delta_y += (n_y - y)/dist

        if dist > umbral:
            self.vx = MAX_SPEED*delta_x/np.min([len(self.neightbors()),10])
            self.vy = MAX_SPEED*delta_y/np.min([len(self.neightbors()),10])
        else:
            self.vx = -MAX_SPEED*delta_x/np.min([len(self.neightbors()),10])
            self.vy = -MAX_SPEED*delta_y/np.min([len(self.neightbors()),10])
        '''
        #donut behavior
        if dist>umbral:
            x_next = wrap(x + BASE_SPEED*delta_x/len(self.neightbors()))
            y_next = wrap(y + BASE_SPEED*delta_y/len(self.neightbors()))
        else:
            x_next = wrap(x - BASE_SPEED*delta_x/len(self.neightbors()))
            y_next = wrap(y - BASE_SPEED*delta_y/len(self.neightbors()))
        '''    

    
    
class SwarmNetwork():

    def __init__(self,home,dataset,start_home = False):

        # Set random intial point:

        x_0 = [] #np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
        y_0 = [] #np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

        # Ensure not to spawn in wall
        for i in range(NUMBER_OF_ROBOTS):
            if start_home == True:
                x, y = home
            else:
                x, y = self.generate_randome_start()
            x_0.append(x)
            y_0.append(y)

        # Velocities random:
        vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
        vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))

        # Agents:
        self.index = np.arange(NUMBER_OF_ROBOTS)
        self.agents = [agent(i,x_0[i],y_0[i],vx[i],vy[i],home,dataset) for i in range(NUMBER_OF_ROBOTS)] # List of agents (own self-position)

        # Adjacency and Laplacian matrix:
        self.Adj = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))
        self.L = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))

        # Set initial topology:
        self.update_Topology()

        # Generate common maps:

        self.global_map = DataSet(dataset,copy=True)
        self.update_map()
        
    
    def generate_randome_start(self):
        x = np.random.randint(low=0, high=ARENA_SIDE_LENGTH)
        y = np.random.randint(low=0, high=ARENA_SIDE_LENGTH)

        if BW_MAP[y,x] == 0:
            return self.generate_randome_start()
        else:    
            return x, y
        
    def state(self):
        return np.array([agent.position() for agent in self.agents])

    def one_step(self, mode = "stop"):

        for agent in self.agents:
            agent.set_mode(mode)
            agent.one_step()
            
            if agent.at_home == True:
                # Update overall map:
                update = self.global_map.merge(agent.dataset,show=False)

        # Update agents neightbourhood:
        self.update_Topology()

    def update_map(self):
        for agent in self.agents:
            if agent.id == 0:
                self.global_map.merge(agent.dataset,show=False)
            else:
                self.global_map.merge(agent.dataset,show=False)
            
    def update_Topology(self):
        
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
                if dist_neighbors[idx] < NEIGHTBORS_SPACE:
                    neightbors_id.append(idx)

            neightbors.append(neightbors_id)

        # Save list of agents as every agent neightbors:
        for i,agent in enumerate(self.agents):
            temp_neightbor = []
            for other_agent in self.agents:
                if other_agent.id in neightbors[i]: 
                    temp_neightbor.append(other_agent)

            # Update agent's neightbour:
            agent.set_neightbors(temp_neightbor)


    # TODO: Update general map every time homing
            

def toggle_mode(event):
    global mode, previous_mode
    if event.key == 'up':
        mode = "random"
    elif event.key == 'down':
        mode = "cluster"
    elif event.key == ' ':
        if mode == "stop":
            mode = previous_mode
        else:
            previous_mode = mode
            mode = "stop"
    elif event.key == 'enter':
        mode = "home"
    elif event.key == 'left':
        mode = "dispersion"


################# PLOT ########################


# Create Map:
map_dataset = MapDataset(ARENA_SIDE_LENGTH, BLOCK_SIZE)

BW_MAP,home = map_dataset.generate_map(walls=False)


# Generate random dataset --> Map discretization with value that agent may infer
DATASET = DataSet(map_dataset)
DATASET.plot_info()


# The agent operate with map and dataset variables to simulate the exploration and data adqusition task.


# Set up the output using map size:
fig = plt.figure(figsize=(BW_MAP.shape[1]/15 , BW_MAP.shape[0]/15), dpi=100)
ax_map = plt.axes([0, 0, 1, 1])  # Adjust position for map
map_plot = ax_map.imshow(BW_MAP, cmap='gray')
ax_map.axis('off')
points, = ax_map.plot([], [], 'bo', lw=0)


# Create swarm:

net = SwarmNetwork(home,map_dataset,start_home=False)
mode = "stop"
previous_mode = "random"


def init():
    points.set_data([], [])
    return points,

def animate(i):

    net.one_step(mode)
    
    # Get points
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]

    points.set_data(x, y)
    print('Step ', i + 1, '/', STEPS, end='\r')

    return points,

fig.canvas.mpl_connect('key_press_event', toggle_mode)

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=STEPS, interval=1, blit=True)

videowriter = animation.FFMpegWriter(fps=60)
#anim.save("..\output.mp4", writer=videowriter)
plt.show()