from imp import load_module
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from map_dataset import MapDataset, DataSet
from dot_swarm import SwarmNetwork
import time

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
NUMBER_OF_ROBOTS  = 10
NEIGHTBORS_SPACE  = 20      # Radius of communication area

# For not printing out all coverd area all the time
global privious_covered_area_ration
privious_covered_area_ration = 0

#TIMER VARIABLES
global timer_reset # Boolean variable for reseting clock each time new behavior is selected
timer_reset=False

initial_time=time.time()
final_time=5000000

simulation_seconds=10

#PLOT VARIABLES

time_axi=[]
value_axi=[]
title=""
y_axi=""

# Control behavior with the keys:
def toggle_mode(event):
    global timer_reset
    timer_reset=True

    global title
    global y_axi

    global mode, previous_mode
    if event.key == 'up':
        mode = "random"
    elif event.key == 'down':
        mode = "cluster"
        title="Cluster Evaluation"
        y_axi="Nº of Clusters"
    elif event.key == ' ':
        if mode == "stop":
            mode = previous_mode
        else:
            previous_mode = mode
            mode = "stop"
    elif event.key == 'enter':
        mode = "home"
        title="Homing Evaluation"
        y_axi="Add y axis here"
    elif event.key == 'right':
        mode = "explore"
        title="Exploration Evaluation"
        y_axi="Covered Area"
    elif event.key == 'left':
        mode = "dispersion"
        title="Dispersion Evaluation"
        y_axi="Covered Area (map cells)"


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

#gets the sorrunding cells taking into account map limits and walls
def get_surrounding_cells(i, j, max_size,occupied_cells):
    # Define the range of j and i coordinates for the surrounding cells
    x_range = range(max(0, i - 1), min(max_size, i + 2))
    y_range = range(max(0, j - 1), min(max_size, j + 2))

    
    for ii in x_range:
        for jj in y_range:
            # Skip the current cell, checks for the map borders
            if ii == i and jj == j:
                continue
            # Skip the current cell, checks for the map walls
            if not DATASET.info[jj][ii].reacheable:
                continue
            occupied_cells.append([ii, jj])
    
    return occupied_cells

def eval_dispersion(net):
    occupied_cells = []
    map_size_limit=int(ARENA_SIDE_LENGTH/BLOCK_SIZE)

    # Compute swarm centroid
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]

    # adds curretnt and sorrounding cells to occupied_cells
    for w in range(len(x)-1):
        i,j=DATASET.index(x[w],y[w])
        get_surrounding_cells(i,j,map_size_limit,occupied_cells)
        occupied_cells.append([i,j])

    # Convert into touples to be able ot use np.unique, if not will count numbers but not pairs of numbers
    pair_tuples = [tuple(pair) for pair in occupied_cells]
    unique_occupied_cells=np.unique(pair_tuples,axis=0)

    print("Ocuppied cells:"+str(len(occupied_cells))+" UniqueOnes:"+str(len(unique_occupied_cells)))
    
    return len(unique_occupied_cells)



# Create Map:
map_dataset = MapDataset(ARENA_SIDE_LENGTH, BLOCK_SIZE)
BW_MAP,home,selected_map = map_dataset.load_map(walls=WALLS_ON)

# Generate random dataset --> Map discretization with value that agent may infer
DATASET = DataSet(map_dataset)
#DATASET.plot_info()


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
mode = "Dispersion"
previous_mode = "Dispersion"


def init():
    points.set_data([], [])
    return points,


def animate(i):
    #timer variables
    global simulation_seconds
    global timer_reset
    global final_time
    global initial_time

    #plotting arrays
    global time_axi
    global value_axi

    global mode
    global privious_covered_area_ration

    net.delete_agent()

    #reset timer each time the behavior is changed
    if timer_reset:
        initial_time=time.time()
        timer_reset=False
        time_axi=[]
        value_axi=[]

    current_time=time.time()-initial_time

    #when set time reached stop simulation
    if simulation_seconds<current_time:
        plt.close()


    net.one_step(mode)

    #call diferent evaluation methods
    if mode=="dispersion":
        cells_ocupied=eval_dispersion(net)
        time_axi.append(current_time)
        value_axi.append(cells_ocupied)

    elif mode=="home":
        print("Insert Home eval")

    elif mode=="cluster":
        n_cluster = eval_cluster(net)
        time_axi.append(current_time)
        value_axi.append(n_cluster)

    elif mode=="explore":
        covered_area_ration = eval_exploration(net)
        if covered_area_ration - privious_covered_area_ration != 0:
            time_axi.append(current_time)
            value_axi.append(covered_area_ration)
            #print('Cover area ration: ', covered_area_ration)
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

# Plotting the eval methods
plt.plot(time_axi, value_axi)
plt.title(title)
plt.xlabel('Time (s)')
plt.ylabel(y_axi)
plt.grid(True)
plt.show()

file_name="saved_plot_data/"+mode+"_"+str(NUMBER_OF_ROBOTS)+"_"+selected_map+".npy"
np.save(file_name, value_axi)