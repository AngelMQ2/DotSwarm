import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import namedtuple
import random
import sys
import time

sys.setrecursionlimit(1500000)

# Data structure:
Data = namedtuple("data",['ts','centroid','value','state'])

#TODO: Add timestamp

class DataSet:

    def __init__(self, map, copy = False):

        self.dim = map.ARENA_SIZE // map.BLOCK_SIZE
        self.K = map.BLOCK_SIZE

        self.figure = None

        info_map = [[None for _ in range(self.dim)] for _ in range(self.dim)]

        cont = 0
        for i in range(self.dim):
            for j in range(self.dim):
                x = int(i*self.K + self.K/2)
                y = int(j*self.K + self.K/2)
                t = time.time()
                value = int(np.random.randint(-20,60))    # TODO: Realistic temperature distribution         
                if map.map[x,y] == 255 and not copy:
                    
                    info_map[i][j] = Data(t,(x,y), value, True)
                    cont += 1
                else:
                    info_map[i][j] = Data(t,(x,y), None, False)

        self.info = info_map
        self.area = cont
        self.last_update = 0

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
    
    '''def plot_info(self):

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
        plt.show(block=True)'''
    
    def plot_info(self):

        def update(ax):
            nonlocal self
            ax.clear()
            
            ax.axis('off')
            ax.set_title('State Plot')  # Set title
            ax.set_xlabel('X')  # Set x-axis label
            ax.set_ylabel('Y')  # Set y-axis label
            ax.invert_yaxis()  # Invert the y-axis
            ax.set_aspect('equal', 'box')

            for i in range(len(self.info)):
                for j in range(len(self.info)):
                    cell = self.info[i][j]
                    color = 'white' if cell.state else 'black'

                    ax.plot(cell.centroid[1], cell.centroid[0], marker='s', markersize=15, color=color)  # Plot square

                    # Annotate the square with the temperature value
                    if cell.value is not None:
                        ax.text(cell.centroid[1], cell.centroid[0], f'{cell.value}', ha='center', va='center')


        if self.figure is None:  # If figure doesn't exist, create a new one
            self.figure = plt.figure(figsize=(8, 6))
        else:  # Clear the previous plot
            self.figure.clear()

        ax = self.figure.gca()  # Get the current axes

        ani = FuncAnimation(self.figure, update(ax), frames=range(10), repeat=True)  # Modify the range as needed
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
    
    def save(self,ts,i,j,value):
        # Save memory:
        old_data = self.info[j][i]
        centroid = self.info[j][i].centroid
        new_data = Data(ts,centroid, value, True)
        self.info[j][i] = new_data
        self.area += 1
        self.last_update = ts
        del old_data

    def store_value(self, x , y, value):
        # Compute index:
        i,j = self.index(x,y)
        # Save memory:
        t = time.time()
        self.save(t,i,j,value)

    def get_value(self,x, y):
        # Compute index:
        i,j = self.index(x,y)
        return self.info[j][i].value
    
    def get_info(self,i,j):
        return self.info[j][i]
    
    def merge(self,source,show=True):

        assert self.dim == source.dim
        update = False

        #if self.area != source.area:
        for i in range(self.dim):
            for j in range(self.dim):
                self_cell = self.get_info(i,j)
                source_cell = source.get_info(i,j)

                # Update in both databases:
                if self_cell.value != source_cell.value:
                    update = True
                    # Compare time-stamp:
                    if self_cell.ts > source_cell.ts:
                        source.save(self_cell.ts,i,j, self_cell.value)

                    elif self_cell.ts < source_cell.ts:
                        self.save(source_cell.ts, i,j, source_cell.value)

        if self.area > source.area:
            self.area = source.area
        else:
            source.area = self.area

        if show and update: self.plot_info()
        return update

class MapDataset:
    def __init__(self, ARENA_SIZE, BLOCK_SIZE):
        self.ARENA_SIZE = ARENA_SIZE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.dim = self.ARENA_SIZE // self.BLOCK_SIZE
        self.map = None
    
    def generate_map(self, walls = True):
        # Create empty map
        map_image = np.ones((self.ARENA_SIZE, self.ARENA_SIZE), dtype=np.uint8)*255

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
        
        '''visited = [[False for _ in range(self.ARENA_SIZE)] for _ in range(self.ARENA_SIZE)]

        # Mark black pixels as visited:
        black_pixels = (np.argwhere(map_image == 0))
        for idx in black_pixels:
            visited[idx[0]][idx[1]] = True

        for i in range(self.ARENA_SIZE):
            for j in range(self.ARENA_SIZE):
                if map_image[i,j] == 255 and not visited[i][j]:
                    # Start exploring neighboring white pixels
                    self.flood_fill(map_image,visited, i, j)

        print('Numero de visitas: ',len(np.argwhere(visited==True)))'''

        # Select home position:
        white_pixels = np.argwhere(map_image == 255)
        home = tuple(white_pixels[np.random.randint(0,len(white_pixels))]) 
        #home = [self.BLOCK_SIZE+self.BLOCK_SIZE/2,self.BLOCK_SIZE+self.BLOCK_SIZE/2]   

        # Plot home block:     
        i = np.min([self.dim, int(home[0] // self.BLOCK_SIZE)])*self.BLOCK_SIZE
        j = np.min([self.dim, int(home[1] // self.BLOCK_SIZE)])*self.BLOCK_SIZE

        map_image[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE] = 150   # Turn block gray, not working
        #print('Home point: ',home)

        self.map = map_image

        return self.map, [home[1],home[0]]    
    
    # Check block connectivity
    def flood_fill(self, map_image, visited, x, y):
            # Define the possible moves (up, down, left, right)
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Set the visited pixel:
            visited[x][y] = True

            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                # Check if the new position is within the image bounds
                if 0 <= new_x < self.ARENA_SIZE and 0 <= new_y < self.ARENA_SIZE:
                    # Check if the neighboring pixel is white and not visited yet
                    if map_image[new_x,new_y] == 255 and not visited[new_x][new_y]:
                        print('Se da')
                        # Recursively explore the neighboring white pixel
                        self.flood_fill(map_image, visited, new_x, new_y)



    def get_map(self):
        return self.map
    
