from imp import load_module
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset, DataSet
from dot_swarm import SwarmNetwork

# Environmen hyperparam:
ARENA_SIDE_LENGTH = 100
BLOCK_SIZE        = 5
WALLS_ON          = True
STEPS             = 10000

# Agent hyperparameters:
MAX_SPEED         = 5e-2
BASE_SPEED        = 5e-3
LIDAR_RANGE       = 5
NUM_RAYS          = 8
SAFE_SPACE        = 2            # Self safe-space, avoid collitions

# Swarm hyperparameters:
NUMBER_OF_ROBOTS  = 20
NEIGHTBORS_SPACE  = 20      # Radius of communication area

# For not printing out all coverd area all the time
global privious_covered_area_ration
privious_covered_area_ration = 0

# Control behavior with the keys:
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
    elif event.key == 'right':
        mode = "explore"
    elif event.key == 'left':
        mode = "dispersion"


def recursive_explore(net, agent_id, visited_list):
    # Deep-first search implementation:
    visited_list[agent_id] = True

    # Get adjacent nodes:
    neightbors_id = np.nonzero(net.Adj[agent_id,:])[0]
    for id in neightbors_id:
        distance = np.linalg.norm([net.agents[agent_id].x - net.agents[id].x,net.agents[agent_id].y - net.agents[id].y])
        if not visited_list[id] and distance < 2*SAFE_SPACE:
            recursive_explore(net, id, visited_list)

def eval_cluster(net):
    # Initialize Deep-first search:
    n = len(net.agents)
    visited_list = [False]*n
    n_cluster = 0

    for agent_id in range(n):
        if not visited_list[agent_id]:
            recursive_explore(net, agent_id, visited_list)
            n_cluster += 1

    return n_cluster


def eval_exploration(net):
    # Look at covered area in exploration:
    current_area = net.global_map.covered_area()
    total_area = DATASET.covered_area()

    return current_area / total_area

def get_centroid(net):
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]
    n = len(x)
    centroid_x = sum(x) / n
    centroid_y = sum(y) / n

    return centroid_x,centroid_y

def eval_dispersion(net):
    max_distance = -1

    # Compute swarm centroid
    centroid_x,centroid_y=get_centroid(net)
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]

    # Get further agent to centroid
    for i in range(len(x)):
        distance = np.linalg.norm([x[i]-centroid_x,y[i]-centroid_y])
        if distance > max_distance:
            max_distance = distance
        
    return max_distance



# Create Map:
map_dataset = MapDataset(ARENA_SIDE_LENGTH, BLOCK_SIZE)
BW_MAP,home = map_dataset.load_map(walls=WALLS_ON)

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

# Evaluation figures:
fig_cl, ax_cl = plt.subplots()
ax_cl.set_xlim(0, STEPS)
ax_cl.set_ylim(0, NUMBER_OF_ROBOTS)  # Adjust ylim based on your data range
#cl_plot, = ax_cl.plot([], [], lw=2)

# Create swarm:
net = SwarmNetwork(home,map_dataset,DATASET, NUMBER_OF_ROBOTS, NEIGHTBORS_SPACE, start_home=True)
mode = "dispersion"
previous_mode = "dispersion"


def init():
    points.set_data([], [])
    return points,



def animate(i):
    global privious_covered_area_ration

    net.one_step(mode)
    
    # Evaluation
    #n_cluster = eval_cluster(net)
    #max_dispersion = eval_dispersion(net)
    covered_area_ration = eval_exploration(net)
    if covered_area_ration - privious_covered_area_ration != 0:
        print('Cover area ration: ', covered_area_ration)
        privious_covered_area_ration = covered_area_ration
    #print('Number of cluster: ', n_cluster)
    #print('Max dispersion distance: ',max_dispersion)

    # Update line plot with number of clusters
    #ax_cl.plot(i,n_cluster)

    # Get state and plot:
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]

    points.set_data(x, y)
    print('Step ', i + 1, '/', STEPS, end='\r')

    return points,

fig.canvas.mpl_connect('key_press_event', toggle_mode)

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=STEPS, interval=1, blit=True)

#anim_cl = FuncAnimation(fig_cl, animate_cluster, init_func=init_cl, frames=STEPS, interval = 1, blit=True)

videowriter = animation.FFMpegWriter(fps=60)
#anim.save("..\output.mp4", writer=videowriter)
plt.show()
