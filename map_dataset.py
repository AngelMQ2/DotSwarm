import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import namedtuple
import random


# Data structure:
Data = namedtuple("data",['centroid','value','state'])

class DataSet:

    def __init__(self, map, copy = False):

        self.dim = map.ARENA_SIZE // map.BLOCK_SIZE
        self.K = map.BLOCK_SIZE

        self.figure = None

        info_map = [[None for _ in range(self.dim)] for _ in range(self.dim)]

        for i in range(self.dim):
            for j in range(self.dim):
                x = int(i*self.K + self.K/2)
                y = int(j*self.K + self.K/2)
                value = int(np.random.randint(-20,60))    # TODO: Realistic temperature distribution         
                if map.map[x,y] == 1 and not copy:
                    info_map[i][j] = Data((x,y), value, True)
                else:
                    info_map[i][j] = Data((x,y), None, False)

        self.info = info_map
    

    # Methode for printing out maps
    def __str__(self):
        str = ""
        for i in range(self.dim):
            str += "\n"
            for j in range(self.dim):   
                if self.info[i][j][1] == None:
                    str += "{:}   ".format(self.info[i][j][2])
                else:
                    str += "{:<7}".format(self.info[i][j][2])
        return str
    
    def plot_info(self):

        if self.figure is None:  # If figure doesn't exist, create a new one
            self.figure = plt.figure(figsize=(8, 6))
        else:  # Clear the previous plot
            self.figure.clear()

        for i in range(len(self.info)):
            for j in range(len(self.info)):
                cell = self.info[i][j]
                color = 'white' if cell.state else 'black'

                plt.plot(cell.centroid[1], cell.centroid[0], marker='s', markersize=15, color=color)  # Plot square

                # Annotate the square with the temperature value
                if cell.value is not None:
                    plt.text(cell.centroid[1], cell.centroid[0], f'{cell.value}', ha='center', va='center')

        ax = self.figure.gca()  # Get the current axes
        ax.axis('off')
        ax.set_title('State Plot')  # Set title
        ax.set_xlabel('X')  # Set x-axis label
        ax.set_ylabel('Y')  # Set y-axis label
        ax.invert_yaxis()  # Invert the y-axis
        ax.set_aspect('equal', 'box')
        plt.show(block=True)

    def index(self, x, y):
        # Compute index:
        i = np.min([self.dim, int(x // self.K)])
        j = np.min([self.dim, int(y // self.K)])
        return i,j
    
    def get_state(self,x,y):
        # Compute index:
        i,j = self.index(x,y)
        
        return self.info[j][i].state
    
    def save(self,i,j,value):
        # Save memory:
        old_data = self.info[j][i]
        centroid = self.info[j][i].centroid
        new_data = Data(centroid, value, True)
        self.info[j][i] = new_data

        del old_data

    def store_value(self, x , y, value):
        # Compute index:
        i,j = self.index(x,y)
        # Save memory:
        self.save(i,j,value)

    def get_value(self,x, y):
        # Compute index:
        i,j = self.index(x,y)
        return self.info[j][i].value
    
    def get_info(self,i,j):
        return self.info[j][i]
    
    def merge(self,source,show=True):

        assert self.dim == source.dim

        update = False
        for i in range(self.dim):
            for j in range(self.dim):
                self_cell = self.get_info(i,j)
                source_cell = source.get_info(i,j)

                if self_cell.state != source_cell.state:
                    # Update in both databases:
                    update = True
                    if self_cell.state == True:
                        source.save(i,j, self_cell.value)
                    elif source_cell.state == True:
                        self.save(i,j, source_cell.value)
                    
        if update and show: self.plot_info()

        return update
        

class MapDataset:
    def __init__(self, ARENA_SIZE, BLOCK_SIZE):
        self.ARENA_SIZE = ARENA_SIZE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.dim = self.ARENA_SIZE // self.BLOCK_SIZE
        self.map = None
    
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

                # Select wall seed:
                x = random.randint(1, self.dim - 2) * self.BLOCK_SIZE
                y = random.randint(1, self.dim - 2) * self.BLOCK_SIZE
                map_image[x:x+self.BLOCK_SIZE, y:y+self.BLOCK_SIZE] = 0

                # Randomly block adjacent cells
                #for _ in range(4):
                direction = random.choice(['up', 'down', 'left', 'right'])
                for j in range(random.randint(0,int(self.dim/2))):
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
        
        # TODO: Check white block full connectivity. Unconnected area with smaller area become black

        '''max_area = 0
        for i in range(self.ARENA_SIZE):
            for j in range(self.ARENA_SIZE):
                if map_image[i,j] == 1:
                    area = self.flood_fill(map_image, i, j)
                    if area < max_area:
                        map_image[i,j] = 0
                    else:
                        max_area = area'''

        # Select home block:
        home_x = np.random.choice(np.argmax(map_image))
        home_y = np.random.choice(np.argmax(map_image))
        map_image[home_x:home_x+self.BLOCK_SIZE, home_y:home_y+self.BLOCK_SIZE] = 0.5   # Turn block gray, not working
        print('Home point: ',home_x,home_y)

        self.map = map_image

        return self.map
    
    # Check block connectivity
    def flood_fill(self, map_image, x, y):
            cont = 0
            if x < 0 or x >= self.ARENA_SIZE or y < 0 or y >= self.ARENA_SIZE or map_image[x, y] != 1:
                return 0
            map_image[x, y] = 2
            cont += self.flood_fill(map_image, x + 1, y)
            cont += self.flood_fill(map_image, x - 1, y)
            cont += self.flood_fill(map_image, x, y + 1)
            cont += self.flood_fill(map_image, x, y - 1)

            return cont

    def get_map(self):
        return self.map
    
