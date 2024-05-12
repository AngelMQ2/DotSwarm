import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import namedtuple
import random
import sys
import time


sys.setrecursionlimit(1500000)


# Data structure to store each map discrete block information: timestamp, discrete block centroid, temperetura, visited (bool) and reacheable block (bool)
Data = namedtuple("Data",['ts','centroid','value','visited','reacheable'])


class DataSet:
    # NOTE: This class generate a matrix which discretize the space given by the black & white map
    # If is not a copy, then the map info is complete available. In case it is a blind_copy, then all the matrix block are set as reacheable and non-visited
    def __init__(self, map, blind_copy = False):  
        # Get dimension from white and black image (map)
        self.arena_size = map.ARENA_SIZE
        self.K = map.BLOCK_SIZE     # Key of the Hash table -> debide continuous (x,y) coordinates by this value to obtain the corresponding place in the matrix
        self.dim = map.ARENA_SIZE // map.BLOCK_SIZE # Dimension of discretazed map

        self.figure = None  # Figure to plot dataset

        # Initilize map dataset. Consist on a graph where each node store the information in Data
        info_map = [[None for _ in range(self.dim)] for _ in range(self.dim)]

        area_cont = 0   # Count number of reacheable blocks (area)
        for i in range(self.dim):
            for j in range(self.dim):
                # Compute block (x,y) centroid, relative to continous space (BW_image)
                x = int(i*self.K + self.K/2)
                y = int(j*self.K + self.K/2)
                t = time.time()                           # Timestep in which data is stored
                value = int(np.random.randint(-20,60))    # Temperature value, randomly chosen     
                
                # Determine wheter is a reacheable block (white) or non-reacheable block (black)
                if map.map[x,y] == 255 and not blind_copy:      # Ensure is not a copy, otherwhise no value is stored as it have to be discovered
                    info_map[i][j] = Data(t,(x,y), value, True, True)
                    area_cont += 1
                else:
                    if blind_copy:
                        info_map[i][j] = Data(t,(x,y), None, False, True)   # Initialize blid graph: non-visited, all reacheable
                    else:
                        info_map[i][j] = Data(t,(x,y), None, False, False)   # Actual cell value for non-reacheable block in the map

        self.info = info_map    # Store matrix
        self.area = area_cont   # Store maximun area (number of block reacheables)
        self.last_update = 0    # Time counter to determine next update

    # Methode for printing out maps
    def __str__(self):
        str = ""
        for i in range(self.dim):
            str += "\n"
            for j in range(self.dim):   
                if self.info[i][j].value == None:
                    if self.info[i][j].reacheable:
                        str += "____   "
                    else:
                        str += "WALL   "
                else:
                    str += "{:<7}".format(self.info[i][j].value)
        return str
    
    # Function to plot in real-time the dataset matrix:
    def plot_info(self):

        def update(ax):
            nonlocal self
            ax.clear()
            
            ax.axis('off')
            ax.set_title('State Plot')  # Set title
            ax.set_xlabel('X')          # Set x-axis label
            ax.set_ylabel('Y')          # Set y-axis label
            ax.invert_yaxis()           # Invert the y-axis
            ax.set_aspect('equal', 'box')

            # Iterate over matrix cells
            for i in range(len(self.info)):
                for j in range(len(self.info)):
                    cell = self.info[i][j]
                    color = 'white' if cell.reacheable else 'black'

                    # Plot in the centroid position a square of color black/white depending in it is reacheable
                    ax.plot(cell.centroid[1], cell.centroid[0], marker='s', markersize=15, color=color)  # Plot square

                    # Annotate the square with the temperature value
                    if cell.value is not None:
                        ax.text(cell.centroid[1], cell.centroid[0], f'{cell.value}', ha='center', va='center')

        # Create figure if needed
        if self.figure is None:  # If figure doesn't exist, create a new one
            self.figure = plt.figure(figsize=(8, 6))
        else:  # Clear the previous plot
            self.figure.clear()

        ax = self.figure.gca()  # Get the current axes
        ani = FuncAnimation(self.figure, update(ax), frames=range(10), repeat=True)  # Modify the range as needed
        plt.show(block=True)

    # Dataset size
    def size(self):
        return (len(self.info),len(self.info[0]))
    
    # Function to obtain the matrix discrete index relative to a concrete continous position (x,y)
    def index(self, x, y):
        # Compute index - Hash table using the key K
        i = np.min([self.dim, int(x // self.K)])
        j = np.min([self.dim, int(y // self.K)])
        return i,j
    
    # Visited value of block corresponding to input position (x,y)
    def get_visited(self, x, y):
        i,j = self.index(x, y)              # Compute index
        return self.info[j][i].visited      # Inverted axes needed because matrix and BW_map has transposes axes
    
    # Return discovered area so far
    def covered_area(self):
        return self.area
    
    # Function to store Data(timestamp, temperature, visited, reacheable) in the index (i,j)
    def save(self, ts, i, j, value, visited, reacheable):
        # Secover previously stored value and get centroid:
        old_data = self.info[j][i]
        centroid = self.info[j][i].centroid # Pick cnetroid
        
        # Generate and store new data:
        new_data = Data(ts, centroid, value, visited, reacheable)   # Generate new named touple with new value
        self.info[j][i] = new_data
        self.last_update = ts          # Last update up-to-date
        if reacheable: self.area += 1  # Increase area score only if saved block is reacheable
        
        del old_data                   # Delete old data - good practice

    # Function to store Data(...) as result of the environment monitoring. Data relative to continous coordinates (x,y)
    def store_value(self, x, y, value, visited, reacheable):
        i,j = self.index(x,y)                           # Compute index
        t = time.time()                                 # Current timestamp for saving
        self.save(t, i, j, value, visited, reacheable)  # Save data

    # Centroid of the block corresponding to current agent's position (x,y)
    def get_centroid(self, x, y):
        i,j = self.index(x, y)               # Compute index
        centroid = self.info[j][i].centroid
        return (centroid[1], centroid[0])    # Inverted axes needed because matrix and BW_map has transposes axes
    
    # Reacheable value of block corresponding to the input position (x,y)
    def get_reacheable(self, x, y):
        i,j = self.index(x, y)               # Compute index
        return self.info[j][i].reacheable    # Inverted axes needed because matrix and BW_map has transposes axes
    
    # Temperature value of block corresponding to the input position (x,y)
    def get_value(self,x, y):
        i,j = self.index(x,y)                # Compute index 
        return self.info[j][i].value         # Inverted axes needed because matrix and BW_map has transposes axes
    
    # Return whole cell related to index (i,j)
    def get_info(self,i,j):
        return self.info[j][i]
    
    # Function to combine information of 2 matrix: self + other (source)
    def merge(self, source, show = False):
        assert self.dim == source.dim
        update = False
        for i in range(self.dim):
            for j in range(self.dim):
                # Extract cell info:
                self_cell = self.get_info(i,j)
                source_cell = source.get_info(i,j)

                # Update in both databases with the newest info per each cell:
                if self_cell != source_cell:    # Determine if same touple
                    update = True
                    # Compare time-stamp:
                    if self_cell.ts > source_cell.ts:   # Self more recent
                        source.save(self_cell.ts, i, j, self_cell.value, self_cell.visited, self_cell.reacheable)
                    elif self_cell.ts < source_cell.ts: # Other more recent
                        self.save(source_cell.ts, i, j, source_cell.value, source_cell.visited, source_cell.reacheable) 

        # Update area counter:
        if self.area > source.area:
            self.area = source.area
        else:
            source.area = self.area

        if show and update: self.plot_info()
        
# NOTE: This class simply generate a black and white image in the form of a 2D numpy array
# There is a parameter to include walls in the map. This walls are randomly generated.
class MapDataset:
    def __init__(self, ARENA_SIZE, BLOCK_SIZE, wall = True):
        self.ARENA_SIZE = ARENA_SIZE                    # Desired size (squared map)
        self.BLOCK_SIZE = BLOCK_SIZE                    # Discretization size
        self.dim = self.ARENA_SIZE // self.BLOCK_SIZE   # Discrete map size

        self.map = None 

    def generate_map(self, walls = True, save = False):
        # Create empty map as a numpy matrix - every cell is one

        map_image = np.ones((self.ARENA_SIZE, self.ARENA_SIZE), dtype=np.uint8)*255

        # Add border walls
        map_image[:, 0:self.BLOCK_SIZE] = 0
        map_image[:, -self.BLOCK_SIZE: self.ARENA_SIZE] = 0
        map_image[0:self.BLOCK_SIZE, :] = 0
        map_image[-self.BLOCK_SIZE:self.ARENA_SIZE, :] = 0

        if walls:   # Add walls
            # Randomly generate walls
            num_blocks = random.randint(5,20)
            for _ in range(num_blocks):

                # Select wall seed - random point in the map from which a wall grows
                x = random.randint(1, self.dim - 2) * self.BLOCK_SIZE
                y = random.randint(1, self.dim - 2) * self.BLOCK_SIZE
                map_image[x:x+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0     # Set chosen block to 0

                # Randomly block adjacent cells
                direction = random.choice(['up', 'down', 'left', 'right'])  # Direction of wall
                for j in range(random.randint(0,int(self.dim/2))):          # Extension of wall
                    if direction == 'up':
                        x_adj = x - j*self.BLOCK_SIZE
                        if x_adj >= 0:
                            map_image[x_adj:x_adj+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0
                    elif direction == 'down':
                        x_adj = x + j*self.BLOCK_SIZE
                        if x_adj < self.ARENA_SIZE:
                            map_image[x_adj:x_adj+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0
                    elif direction == 'left':
                        y_adj = y - j*self.BLOCK_SIZE
                        if y_adj >= 0:
                            map_image[x:x+self.BLOCK_SIZE, y_adj:y_adj+self.BLOCK_SIZE] = 0
                    elif direction == 'right':
                        y_adj = y + j*self.BLOCK_SIZE
                        if y_adj < self.ARENA_SIZE:
                            map_image[x:x+self.BLOCK_SIZE, y_adj:y_adj+self.BLOCK_SIZE] = 0
        
        # Select home position:
        white_pixels = np.argwhere(map_image == 255)
        home = tuple(white_pixels[np.random.randint(0,len(white_pixels))]) 
        # Plot home block:     
        i = np.min([self.dim, int(home[0] // self.BLOCK_SIZE)])*self.BLOCK_SIZE
        j = np.min([self.dim, int(home[1] // self.BLOCK_SIZE)])*self.BLOCK_SIZE
        map_image[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE] = 150   # Turn block gray

        # Save BW image
        self.map = map_image
        
        # Save as np file
        if save:
            np.save(f"map_{int(time.time())}", self.map)

        return self.map, [home[1],home[0]]  
    
    def load_map(self, walls = True):
        print("\nPut in the file name, or chose map 1, 2, or 3 with numbers.\n"
              + "Press \"r\" for new random map")
        file_name = input()
        if file_name == "1":
            print("load map 1")
            self.map = np.load("map_1_easy.npy")
        elif file_name == "2":
            print("load map 2")
            self.map = np.load("map_2_medium.npy")
        elif file_name == "3":
            print("load map 3")
            self.map = np.load("map_3_hard.npy")
        elif file_name == "r":
            print("generate new map")
            return self.generate_map(walls)[0],self.generate_map(walls)[1], file_name
        elif file_name == "rs": # generate new map and save
            print("generate new map and save")
            return self.generate_map(walls, save =  True)[0], self.generate_map(walls, save =  True)[1],file_name
        else:
            self.map = np.load(file_name)
        return self.map, [np.where(self.map == 150)[1][0], np.where(self.map == 150)[0][0]], file_name

    def get_map(self):
        return self.map
    

