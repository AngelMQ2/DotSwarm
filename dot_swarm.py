import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset, DataSet

# Environmen hyperparam:
ARENA_SIDE_LENGTH = 100
BLOCK_SIZE        = 5
STEPS             = 1000
MAX_SPEED         = 5e-2
BASE_SPEED        = 5e-3

# Swarm hyperparameters:
NUMBER_OF_ROBOTS  = 20
NEIGHTBORS_SPACE = 20      # Radius of communication area
SAFE_SPACE = 2            # Self safe-space, avoid collitions


# Make the environment toroidal 
def wrap(z):    
    return z % ARENA_SIDE_LENGTH

class agent:
    def __init__(self, id, x, y, vx, vy, dataset):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        self.N = None 

        # Create agent's dataset:
        self.dataset = DataSet(dataset,copy=True)

      
    def position(self):
        return np.array([self.x, self.y])

    def set_position(self, new_x, new_y):
        x,y = self.avoid_collision(new_x,new_y)
        self.x = x
        self.y = y

        self.monitoring()

    def neightbors(self):
        return self.N

    def set_neightbors(self,neightbors):
        self.N = neightbors

    def monitoring(self):
        if not self.dataset.get_state(self.x, self.y):
            #if self.id == 0:
            #    self.dataset.plot_info()
            value = DATASET.get_value(self.x, self.y)
            self.dataset.store_value(self.x, self.y, value)

    def forward(self):
        x_next = wrap(self.x + self.vx)
        y_next = wrap(self.y + self.vy)
        self.set_position(x_next, y_next)
        return x_next, y_next
    
    def stop(self):
        self.set_position(self.x, self.y)
        return self.x, self.y

    def avoid_collision(self,x,y):
        new_x = np.copy(x)
        new_y = np.copy(y)

        if new_x != self.x and new_y != self.y:
            for n in self.N:
                distance = np.linalg.norm(self.position() - n.position()) + 0.001
                if distance < SAFE_SPACE:
                    # Calculate the adjustment vector
                    adjustment = [float(diff) for diff in (self.position() - n.position())]
                    adjustment /= distance  # Normalize to unit vector
                    adjustment *= (SAFE_SPACE - distance) / 2  # Scale by the amount to adjust
                    new_x += adjustment[0]
                    new_y += adjustment[1]
        
        return new_x, new_y
    
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
        
        if self.neightbors() == []:
            delta_x += MAX_SPEED #*np.random.uniform(0,1)
            delta_y += MAX_SPEED #*np.random.uniform(0,1)

        x_next = wrap(x + delta_x)
        y_next = wrap(y + delta_y)
        
        self.set_position(x_next,y_next)
        return x_next, y_next
    
    
class SwarmNetwork():

    def __init__(self,dataset):

        # Set random intial point:
        x_0 = [] #np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
        y_0 = [] #np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

        # Ensure not to spawn in wall
        for i in range(NUMBER_OF_ROBOTS):
            x, y = self.generate_randome_start()
            x_0.append(x)
            y_0.append(y)

        # Velocities random:
        vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
        vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))

        # Agents:
        self.index = np.arange(NUMBER_OF_ROBOTS)
        self.agents = [agent(i,x_0[i],y_0[i],vx[i],vy[i],dataset) for i in range(NUMBER_OF_ROBOTS)] # List of agents (own self-position)

        # Adjacency and Laplacian matrix:
        self.Adj = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))
        self.L = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))

        # Set initial topology:
        self.update_Topology()

        # Generate common maps:
        self.global_map = DataSet(dataset,copy=True)
        self.update_map()
        #self.global_map.plot_info()
    
    def generate_randome_start(self):
        x = np.random.randint(low=0, high=ARENA_SIDE_LENGTH)
        y = np.random.randint(low=0, high=ARENA_SIDE_LENGTH)

        if BW_MAP[y,x] == 0:
            return self.generate_randome_start()
        else:    
            return x, y
        
    def state(self):
        return np.array([agent.position() for agent in self.agents])

    def one_step(self, mode = "random"):
        x = []
        y = []
        for agent in self.agents:
            if mode == "random":
                _x,_y = agent.forward()
            elif mode == "cluster":
                _x,_y = agent.cluster()
            else:   # Do nothing
                _x,_y = agent.stop()

            x.append(_x)
            y.append(_y)

        self.update_Topology()
        self.update_map()

    def update_map(self):
        for agent in self.agents:
            if agent.id == 0:
                updated = self.global_map.merge(agent.dataset,show=False)
            else:
                updated = self.global_map.merge(agent.dataset,show=False)

            #if updated: self.global_map.plot_info() # print(self.global_map)
            
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

    # TODO: Function to plot shared blind information map

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


################# PLOT ########################


# Create Map:
map_dataset = MapDataset(ARENA_SIDE_LENGTH, BLOCK_SIZE)
BW_MAP = map_dataset.generate_map(walls=True)

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

#map_dataset.plot_info()

# Set up the output (1024 x 768):
#fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
#ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
#points, = ax.plot([], [], 'bo', lw=0, )



# Create swarm:
net = SwarmNetwork(map_dataset)
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