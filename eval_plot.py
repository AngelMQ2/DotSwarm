import os
import numpy as np
import matplotlib.pyplot as plt

def custom_sort_key(file_name):
    map_name = file_name.split("_")[-1].split(".")[0]
    if map_name == "nw":
        return 0
    else:
        return int(map_name)

def plot_npy_objects(mode, time,map_name):
    folder_path = "saved_plot_data"
    npy_files = [file for file in os.listdir(folder_path) if file.startswith(mode) and file.endswith(f"_{map_name}.npy")]

    if len(npy_files) != 3:
        print("Error: There should be 3 npy files starting with the provided keyword.")
        return

    plt.figure(figsize=(10, 6))


    for npy_file in sorted(npy_files, key=lambda x: int(x.split("_")[1])):
        full_path = os.path.join(folder_path, npy_file)
        data = np.load(full_path)
        if mode=="home"or mode=="explore":
            data=data*100
        time_array = np.linspace(0, time, len(data))  # Creating time array from 0 to indicated time
        plt.plot(time_array, data, label= npy_file.split("_")[1] )

    if mode=="dispersion":
        title="Dispersion Evaluation"
        y_axi="Covered Area (u²)"

    elif mode=="home":
        title="Homing Evaluation"
        y_axi = "Agents at home (%)"

    elif mode=="cluster":
        title="Cluster Evaluation"
        y_axi="Nº of Clusters"

    elif mode=="explore":
        title="Exploration Evaluation"
        y_axi="Explored Area (%)"

    plt.xlabel("Time (s)")
    plt.ylabel(y_axi)
    plt.title(title)
    plt.legend(title="Nº robots")
    plt.grid(True)
    plt.show()

def plot_npy_objects_map(mode, num_agents, time):
    folder_path = "saved_plot_data"
    npy_files = [file for file in os.listdir(folder_path) if file.startswith(f"{mode}_{num_agents}")]

    sorted_files = sorted(npy_files, key=custom_sort_key)


    if len(npy_files) != 3:
        print("Error: There should be 3 npy files starting with the provided mode and number of robots.")
        return

    plt.figure(figsize=(10, 6))

    for npy_file in sorted_files:
        full_path = os.path.join(folder_path, npy_file)
        data = np.load(full_path)
        if mode=="home" or mode=="explore":
            data=data*100
        map_name = npy_file.split("_")[-1].split(".")[0]  # Extracting the map name
        if map_name=="nw":
            map_name="Empty Map"
        elif map_name=="1":
            map_name="Easy Map"
        elif map_name=="3":
            map_name="Hard Map"
        time_array = np.linspace(0, time, len(data))  # Creating time array from 0 to indicated time
        plt.plot(time_array, data, label=map_name)

    if mode == "dispersion":
        title = "Dispersion Evaluation (Nº agents:"+str(num_agents)+")"
        y_axi="Covered Area (u²)"

    elif mode == "home":
        title = "Homing Evaluation (Nº agents:"+str(num_agents)+")"
        y_axi = "Agents at home (%)"

    elif mode == "cluster":
        title = "Cluster Evaluation (Nº agents:"+str(num_agents)+")"
        y_axi = "Nº of Clusters"

    elif mode == "explore":
        title = "Exploration Evaluation (Nº agents:"+str(num_agents)+")"
        y_axi="Explored Area (%)"

    plt.xlabel("Time (s)")
    plt.ylabel(y_axi)
    plt.title(title)
    plt.legend(title="Map Name")
    plt.grid(True)
    plt.show()

mode = "explore"  # Change this to "dispersion", "explore", "home", or "cluster" as needed
time = 100  # Change this to the desired time
num_agents=20
map_name="1"
plot_npy_objects(mode, time,map_name)
plot_npy_objects_map(mode, num_agents, time)
