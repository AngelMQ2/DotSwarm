import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import namedtuple
import random

# Data structure:
Data = namedtuple("data",['centroid','value','occupation'])

class MapDataset:
    def __init__(self, ARENA_SIZE, BLOCK_SIZE):
        self.ARENA_SIZE = ARENA_SIZE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.K = self.ARENA_SIZE // self.BLOCK_SIZE
        self.map = None
        self.info = None
        
    def generate_map(self, walls = True):
        # Create empty map
        map_image = np.ones((self.ARENA_SIZE, self.ARENA_SIZE), dtype=np.uint8)

        # Add border walls
        map_image[:, 0:self.BLOCK_SIZE] = 0
        map_image[:, -self.BLOCK_SIZE: self.ARENA_SIZE] = 0
        map_image[0:self.BLOCK_SIZE, :] = 0
        map_image[-self.BLOCK_SIZE:self.ARENA_SIZE, :] = 0

        if walls:

            # Randomly generate walls
            num_blocks = random.randint(5,20)
            for _ in range(num_blocks):
                x = random.randint(1, self.K - 2) * self.BLOCK_SIZE
                y = random.randint(1, self.K - 2) * self.BLOCK_SIZE
                map_image[x:x+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0

                # Randomly block adjacent cells
                for _ in range(4):
                    direction = random.choice(['up', 'down', 'left', 'right'])
                    for j in range(random.randint(1,self.K)):
                        if direction == 'up':
                            x_adj = x - self.BLOCK_SIZE
                            if x_adj >= 0:
                                map_image[x_adj:x_adj+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0
                        elif direction == 'down':
                            x_adj = x + self.BLOCK_SIZE
                            if x_adj < self.ARENA_SIZE:
                                map_image[x_adj:x_adj+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0
                        elif direction == 'left':
                            y_adj = y - self.BLOCK_SIZE
                            if y_adj >= 0:
                                map_image[x:x+self.BLOCK_SIZE, y_adj:y_adj+self.BLOCK_SIZE] = 0
                        elif direction == 'right':
                            y_adj = y + self.BLOCK_SIZE
                            if y_adj < self.ARENA_SIZE:
                                map_image[x:x+self.BLOCK_SIZE, y_adj:y_adj+self.BLOCK_SIZE] = 0
        
        self.map = map_image

        # Generate info map:
        self.generate_info()

        return self.map

    def generate_info(self):
        # Create empy info map:
        info_map = [[None for _ in range(self.K)] for _ in range(self.K)]

        for i in range(self.K):
            for j in range(self.K):
                x = int(i*self.BLOCK_SIZE + self.BLOCK_SIZE/2)
                y = int(j*self.BLOCK_SIZE + self.BLOCK_SIZE/2)
                value = np.random.uniform(-20,60)    # TODO: Realistic temperature distribution         
                if self.map[x,y] == 1:
                    info_map[i][j] = Data((x,y), value, True)
                else:
                    info_map[i][j] = Data((x,y), None, False)

        self.info = info_map
    
    def generate_blind_copy(self,reference_map):
        self.map = np.copy(reference_map.map)

        # Create blind info map:
        info_map = [[None for _ in range(self.K)] for _ in range(self.K)]

        for i in range(self.K):
            for j in range(self.K):
                x = int(i*self.BLOCK_SIZE + self.BLOCK_SIZE/2)
                y = int(j*self.BLOCK_SIZE + self.BLOCK_SIZE/2)
                info_map[i][j] = Data((x,y), None, False)
        
        self.info = info_map

    def get_map(self):
        return self.map
    
    def get_info_map(self):
        return self.info
    
    def plot_info(self):
        plt.figure(figsize=(8, 6))
        for i in range(len(self.info)):
            for j in range(len(self.info[i])):
                cell = self.info[i][j]
                color = 'white' if cell.occupation else 'black'
                plt.plot(cell.centroid[1], cell.centroid[0], marker='s', markersize=15, color=color)  # Plot square

                # Annotate the square with the temperature value
                if cell.value is not None:
                    plt.text(cell.centroid[1], cell.centroid[0], f'{cell.value:.2f}', ha='center', va='center')


        plt.title('Occupation Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        # Invert the axes
        #plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

'''ARENA_SIZE = 100
BLOCK_SIZE = 5
STEPS = 1000
map_dataset = MapDataset(ARENA_SIZE, BLOCK_SIZE)
map_image = map_dataset.generate_map(walls=True)
map_dataset.plot_info()

# Set up the output using map size:
fig = plt.figure(figsize=(map_image.shape[1]/15 , map_image.shape[0]/15), dpi=100)

ax_map = plt.axes([0, 0, 1, 1])  # Adjust position for map

map_plot = ax_map.imshow(map_image, cmap='gray')
ax_map.axis('off')
points, = ax_map.plot([], [], 'bo', lw=0)

# Create swarm:
# net = SwarmNetwork()
# mode = "stop"
# previous_mode = "random"

# Define swarm animation functions here

def init():
    return map_plot,

def animate(i):
    # Update swarm
    # net.one_step(mode)
    # p = net.state()

    # Dummy data for demonstration
    p = np.random.rand(20, 2) * ARENA_SIZE

    x = p[:, 0]
    y = p[:, 1]

    points.set_data(x, y)
    
    print('Step ', i + 1, '/', STEPS, end='\r')

    return map_plot

# Define key press event handler here

# fig.canvas.mpl_connect('key_press_event', toggle_mode)

#anim = FuncAnimation(fig, animate, init_func=init, frames=10, interval=200, blit=True)

# videowriter = animation.FFMpegWriter(fps=60)
# anim.save("..\output.mp4", writer=videowriter)

plt.show()'''
