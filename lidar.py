import numpy as np
import matplotlib.pyplot as plt

# Constants
LIDAR_RANGE = 10  # Range of the LIDAR
NUM_RAYS = 4  # Number of LIDAR rays
AGENT_POS = (4, 4)  # Agent position (row, column) in the environment

# Example environment (black and white image)
environment = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

def lidar_scan(environment, agent_pos, lidar_range, num_rays):
    lidar_readings = []
    phi = np.pi/3
    for angle in np.linspace(phi,phi +  2*np.pi, num_rays, endpoint=False):
        dx = np.cos(angle)
        dy = np.sin(angle)
        ray = []
        for step in range(lidar_range):
            x = int(agent_pos[0] + dx * step)
            y = int(agent_pos[1] + dy * step)
            if 0 <= x < environment.shape[0] and 0 <= y < environment.shape[1]:
                ray.append(environment[x, y])
            else:
                ray.append(0)  # If ray goes out of bounds, consider it as hitting an obstacle
        lidar_readings.append(ray)
    return lidar_readings

# Perform LIDAR scan
lidar_data = lidar_scan(environment, AGENT_POS, LIDAR_RANGE, NUM_RAYS)

# Plot the environment with agent and LIDAR rays
plt.figure(figsize=(8, 8))
plt.imshow(environment, cmap='gray', origin='lower')
plt.title('Environment with Agent and LIDAR Rays')
plt.plot(AGENT_POS[1], AGENT_POS[0], 'ro')  # Plot agent position
for angle_idx, ray in enumerate(lidar_data):
    for step, val in enumerate(ray):
        x = AGENT_POS[1] + step * np.cos(angle_idx * (2*np.pi / NUM_RAYS))
        y = AGENT_POS[0] + step * np.sin(angle_idx * (2*np.pi / NUM_RAYS))
        if val == 0:
            plt.plot(x, y, 'r.')  # Plot obstacle encountered by LIDAR
        else:
            plt.plot(x, y, 'b.')  # Plot free space encountered by LIDAR


# Plot the LIDAR readings
plt.figure(figsize=(8, 5))
for i, ray in enumerate(lidar_data):
    plt.plot(range(LIDAR_RANGE), ray, label=f'Ray {i+1}')
plt.xlabel('Distance')
plt.ylabel('Obstacle (0) / Free space (1)')
plt.title('LIDAR Readings')
plt.legend()
plt.show()
