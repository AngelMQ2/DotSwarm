from re import M
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset

# Environmen hyperparam:
ARENA_SIDE_LENGTH = 100
BLOCK_SIZE        = 5
STEPS             = 1000
MAX_SPEED         = 5e-2
BASE_SPEED        = 1e-3

# Swarm hyperparameters:
NUMBER_OF_ROBOTS  = 50 #50
NUMBER_OF_NEIGHTBORS = 5
NEIGHTBORS_SPACE = 100      # Radius of communication area
SAFE_SPACE = 2.5            # Self safe-space, avoid collitions

assert NUMBER_OF_ROBOTS >= NUMBER_OF_NEIGHTBORS

# Make the environment toroidal 
def wrap(z):    
    return z % ARENA_SIDE_LENGTH

class agent:
    def __init__(self, id, x, y, vx, vy, map_dataset):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        self.N = None 

        ## needed lookup map may delet later
        self.map_dataset = map_dataset
        # Create local map:
        self.map = MapDataset(ARENA_SIDE_LENGTH,BLOCK_SIZE)
        self.map.generate_blind_copy(map_dataset)
        print(f"x {x}, y {y}")
        print(f"value {map_dataset.get_value_at(int(y*2/10), int(x*2/10))} x {int(x*2/10)} y {int(y*2/10)}")

    def position(self):
        return np.array([self.x, self.y])

    def set_position(self, new_x, new_y):
        x,y = self.avoid_collision(new_x,new_y)
        self.x = x
        self.y = y

    def neightbors(self):
        return self.N

    def set_neightbors(self,neightbors):
        self.N = neightbors

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

        # avoid walls
        if self.map_dataset.get_value_at(int(new_y*2/10), int(new_x*2/10)) == None or self.map_dataset.get_value_at(int((new_y+SAFE_SPACE)*2/10), int((new_x+SAFE_SPACE)*2/10)) == None:
            self.vx = -self.vx
            self.vy = -self.vy
            new_x += self.vx
            new_y += self.vy

        if new_x != self.x and new_y != self.y:
            for n in self.N:
                distance = np.linalg.norm(self.position() - n.position())
                if distance < SAFE_SPACE:
                    # Calculate the adjustment vector
                    adjustment = self.position() - n.position()
                    adjustment /= distance  # Normalize to unit vector
                    adjustment *= (SAFE_SPACE - distance) / 2  # Scale by the amount to adjust
                    new_x += adjustment[0]
                    new_y += adjustment[1]
        
        return new_x, new_y 
    
    def cluster(self,state):
        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0

        [x,y] = self.position()
        
        # Control law:
        for n in self.neightbors():
            [n_x, n_y] = state[n.id]
            delta_x += n_x - x
            delta_y += n_y - y
        
        x_next = wrap(x + BASE_SPEED*delta_x)
        y_next = wrap(y + BASE_SPEED*delta_y)
        
        self.set_position(x_next,y_next)
        return x_next, y_next

    def update_map(self):
        value = self.map_dataset.get_value_at(int(y*2/10), int(x*2/10))

    

    
class SwarmNetwork():

    def __init__(self,map_dataset):
        # Set random intial point:
        x_0 = []
        y_0 = []

        # ensure not to spawn in wall
        for i in range(NUMBER_OF_ROBOTS):
            x, y = self.generate_randome_start(map_dataset)
            x_0.append(x)
            y_0.append(y)

        # Velocities random:
        vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
        vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))

        # Agents:
        self.index = np.arange(NUMBER_OF_ROBOTS)
        self.agents = [agent(i,x_0[i],y_0[i],vx[i],vy[i],map_dataset) for i in range(NUMBER_OF_ROBOTS)] # List of agents (own self-position)

        # Adjacency and Laplacian matrix:
        self.Adj = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))
        self.L = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))

        # Set initial topology:
        self.update_Topology()

        # Generate common maps:
        self.shared_map = MapDataset(ARENA_SIDE_LENGTH,BLOCK_SIZE)
        self.shared_map = self.shared_map.generate_blind_copy(map_dataset)

    def generate_randome_start(self, map_dataset):
        x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH)
        y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH)
        try: 
            if map_dataset.get_value_at(int(y*2/10), int(x*2/10)) == None or map_dataset.get_value_at(int((y+SAFE_SPACE)*2/10), int((x+SAFE_SPACE)*2/10)) == None:
                return self.generate_randome_start(map_dataset)
            else:
                return x, y
        except:
            return self.generate_randome_start(map_dataset)

    def state(self):
        return np.array([agent.position() for agent in self.agents])

    def update_position(self,delta_x,delta_y):
        for i,agent in enumerate(self.agents):
            agent.set_position(delta_x[i],delta_y[i])

    def one_step(self, mode = "random"):
        
        x = []
        y = []
        for agent in self.agents:
            if mode == "random":
                _x,_y = agent.forward()
            elif mode == "cluster":
                _x,_y = agent.cluster(self.state())
            else:   # Do nothing
                _x,_y = agent.stop()

            x.append(_x)
            y.append(_y)

        # Update all agent position at once to avoid troubles in the algorithm 
        # Each agent has made its decision individually
        #self.update_position(x,y)
        self.update_Topology()

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

        #TODO: Check unconected subgraph
            

            
        # Double neightbour correlation:    To be neightbors, 2 agents must be neightbors respectively
        '''for agent in self.agents:
            c_neight = agent.neightbors()

            for j,n in enumerate(c_neight):
                if agent.id not in self.agents[n].neightbors():
                    c_neight[j] = agent.id    # Self-reference, agent connected with itself
                else:
                    self.Adj[agent.id, n] = 1
                    self.Adj[n, agent.id] = 1 
            
            # Update agent's neightbor:
            agent.set_neightbors(c_neight)'''


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
map_image = map_dataset.generate_map(walls=True)
print(map_dataset)

# Set up the output using map size:
fig = plt.figure(figsize=(map_image.shape[1]/15 , map_image.shape[0]/15), dpi=100)
ax_map = plt.axes([0, 0, 1, 1])  # Adjust position for map
map_plot = ax_map.imshow(map_image, cmap='gray')
ax_map.axis('off')
points, = ax_map.plot([], [], 'bo', lw=0)

map_dataset.plot_info()

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