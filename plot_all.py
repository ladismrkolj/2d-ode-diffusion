import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Step 2 & 3
plt.rcParams['text.usetex']       = False
plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus']= False
print("DejaVu Sans at:", fm.findfont('DejaVu Sans'))
import matplotlib.animation as animation
import re
import math
import scienceplots
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.geometry import LineString


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def halfway_between(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the midpoint
    halfway_x = (x1 + x2) / 2
    halfway_y = (y1 + y2) / 2

    return (halfway_x, halfway_y)

def generate_convex_hull(points):
    """
    Computes the convex hull of a set of points.

    Parameters:
    points (list of tuples): List of (x, y) coordinates.

    Returns:
    list of tuples: Points on the convex hull.
    """
    points_array = np.array(points)
    hull = ConvexHull(points_array)
    hull_points = points_array[hull.vertices]
    return hull_points

def offset_polygon(polygon, offset_distance):
    """
    Offsets a polygon outward by a given distance.

    Parameters:
    polygon (shapely.geometry.Polygon): The original polygon.
    offset_distance (float): Distance to offset outward.

    Returns:
    shapely.geometry.Polygon: The offset polygon.
    """
    return polygon.buffer(offset_distance)

def generate_border_points(polygon, spacing):
    """
    Generates points along the border of a polygon at specified intervals.

    Parameters:
    polygon (shapely.geometry.Polygon): The polygon to generate points on.
    spacing (float): Spacing between generated points.

    Returns:
    list of tuples: Points on the border of the polygon.
    """
    # Convert polygon to a list of LineStrings (edges)
    exterior_line = LineString(polygon.exterior.coords)
    
    # Calculate the total length of the boundary
    total_length = exterior_line.length
    
    # Generate points along the boundary
    border_points = []
    num_points = int(np.ceil(total_length / spacing))
    for i in range(num_points):
        distance = i * spacing
        if distance > total_length:
            distance = total_length
        point = exterior_line.interpolate(distance)
        border_points.append((point.x, point.y))
    
    return border_points

def generate_grid_from_real_points(input_points_main, input_points_aux2, max_distance):
    temp_grid = []
    grid = []
    input_points = []

    for i in input_points_main:
        x, y = i
        grid.append((x,y,"main",[]))
        input_points.append(i)
    print(len(input_points))
        
    for i in input_points_aux2:
        x, y = i
        grid.append((x,y,"aux2",[]))
        input_points.append(i)
    print(len(input_points))
        
    num_points = len(input_points)
    new_points = []
    # Initialize the grid with empty neighbor lists
    for i in range(num_points):
        x, y = input_points[i]
        # Determine neighbors for point i
        for j in range(i+1, num_points):
            dist = calculate_distance(input_points[i], input_points[j])
            if dist <= max_distance:
                np_coords = halfway_between(input_points[i], input_points[j])
                new_points.append((np_coords[0], np_coords[1], "aux", [i,j]))
                np_index = num_points + len(new_points) - 1
                grid[i][3].append(np_index)
                grid[j][3].append(np_index)
                
    print(f"len(new_points)={len(new_points)}")
    
    # Generate convex hull
    hull_points = generate_convex_hull(input_points)
    hull_polygon = Polygon(hull_points)

    # Offset polygon and generate border points
    border_offset = 1
    border_spacing = 3

    offseted_polygon = offset_polygon(hull_polygon, border_offset)
    raw_border_points = generate_border_points(offseted_polygon, border_spacing)
    #plot_results(input_points, hull_points, offseted_polygon, raw_border_points)
    border_points = [(x, y, "aux", []) for x, y in raw_border_points]
    num_border_points=len(border_points)
    num_new_points = len(new_points)
    print(num_border_points)

    for i in range(num_border_points):
        # Determine neighbors for point i
        np_index = num_points + num_new_points + i
        for j in range(num_points):
            dist = calculate_distance(input_points[j], border_points[i])
            if dist <= max_distance:
                grid[j][3].append(np_index)
                border_points[i][3].append(j)
        for j in range(num_new_points):
            dist = calculate_distance(new_points[j], border_points[i])
            if dist <= max_distance:
                new_points[j][3].append(np_index)
                border_points[i][3].append(num_points + j)
        
    print(len(grid))
    grid.extend(new_points)
    print(len(grid))
    grid.extend(border_points)
    print(len(grid))
    return grid, border_points

def plot_hex_grid(hex_grid):
    main_points = [point for point in hex_grid if point[2] == 'main']
    aux_points = [point for point in hex_grid if point[2] == 'aux']
    aux2_points = [point for point in hex_grid if point[2] == 'aux2']
    
    print(len(main_points))
    print(len(aux_points))
    print(len(aux2_points))
    
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.figure(figsize=(5,5), dpi = 300)

    
    # Plot aux points
    if aux_points:
        aux_x, aux_y = zip(*[(x, y) for x, y, _, _ in aux_points])
        plt.scatter(aux_x, aux_y, color='blue', label='ZuCT')
    
    # Plot main points
    if main_points:
        main_x, main_y = zip(*[(x, y) for x, y, _, _ in main_points])
        plt.scatter(main_x, main_y, color='yellow', label='Akson')
        
    if aux2_points:
        main_x, main_y = zip(*[(x, y) for x, y, _, _ in aux2_points])
        plt.scatter(main_x, main_y, color='red', label='Kapilara')
    
    # Label each point with its index
    #for i, (x, y, _, _) in enumerate(hex_grid):
    #    plt.text(x, y, str(i), fontsize=9, ha='right')
    
    #plt.title('Hexagonal Grid with Main and Aux Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X / $\mathrm{μm}$')
    plt.ylabel('Y / $\mathrm{μm}$')
    plt.savefig(f'hex_grid.png', dpi=150)
    #plt.show()

import json

def load_json_to_tuples(file_path):
    """
    Reads JSON data from a file and converts it to a list of tuples.
    
    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    list of tuples: Converted data.
    """
    with open(file_path, 'r') as file:
        data_list = json.load(file)
    
    # Convert list of lists to list of tuples
    tuple_list = [tuple(item) for item in data_list]
    
    return tuple_list


def import_results(file_path):
    data = pd.read_csv(file_path)

    columns = data.columns[1:]

    proc_columns = []
    for col in columns:
        str_col = col[2:-1]
        spl_col = str_col.split(', ')
        flt_col = [float(i) for i in spl_col]
        proc_columns.append(flt_col)

    coords = np.array(proc_columns)
    values = np.array(data.iloc[:, 1:].values)
    times = np.array(data.iloc[:, 0].values)
    return coords, values, times

def mirror(coords, values):
    def mirror_y_axis(x, y):
        return -x, y
    def rotate_x_deg(x, y, a):
        xdot = x * math.cos(a) + y * math.sin(a)
        ydot = -x * math.sin(a) + y * math.cos(a)
        return xdot, ydot
    # Function to check if a point is within the triangle
    def in_triangle(x, y):
        return (0 <= x <= y/math.sqrt(3)*1.02) and (x*math.sqrt(3)*0.98 <= y)

    indices_to_delete=[]
    for i, point in enumerate(coords):
        if not in_triangle(*point):
            indices_to_delete.append(i)
            
    print(coords.shape, values.shape)
    coords = np.delete(coords, indices_to_delete, 0)
    values = np.delete(values, indices_to_delete, 1)
    print(coords.shape, values.shape)
    #fill the missing dots:
    indces_to_copy_left=[]
    for i, point in enumerate(coords):
        if point[0] == 3.414213562373095:
            indces_to_copy_left.append(i)
        if point[0] == 0 and point[1] == 0:
            zero_index = i

    new_coords = []
    new_values = []
    for i in indces_to_copy_left:
        new_coords.append([[0.0, coords[i,1]]])
        new_values.append([[a] for a in values[:,i]])

    new_coords.append([[0.0, 5.913591357920932/2]])
    new_values.append([[a] for a in values[:,zero_index]])
        
    for i in range(len(new_coords)):
        coords = np.append(new_coords[i], coords, axis=0)
        values = np.append(new_values[i], values, axis=1)
    print(coords.shape, values.shape)
    coords = list(coords)

    mirrored_over_y_coords = [(mirror_y_axis(x,y)) for x,y in coords]

    coords.extend(mirrored_over_y_coords)

    rotated_coords = []
    for a in [math.pi/3, 2*math.pi/3, 3*math.pi/3, 4*math.pi/3, 5*math.pi/3]:
        temp_coords = [(rotate_x_deg(x,y, a)) for x,y in coords]
        rotated_coords.extend(temp_coords)
        
    coords.extend(rotated_coords)
    coords = np.array(coords)

    # Print initial lengths for verification
    print("Initial length of second row:", len(values[1]))

    # Replicate each row's elements in an interleaved manner (ababab)
    replicated_values = np.empty((values.shape[0], values.shape[1] * 12))

    # Fill the replicated_values array with the interleaved pattern
    for i in range(values.shape[1]):
        replicated_values[:, i::values.shape[1]] = values[:, i][:, np.newaxis]

    # Print length after replication for verification
    print("Length of second row after replication:", len(replicated_values[1]))

    values = replicated_values
    return coords, values

    # Define the times of interest

#times_of_interest = [0.0, 100.0, 500.0, 1000.0]
def plot_time_series(toi, coords, values, times, label):

    times_of_interest = toi
    # Extract the time values from the data
    time_values = times

    # Find the indices that correspond to the desired times
    frames_to_save = [np.argmin(np.abs(time_values - t)) for t in times_of_interest]

    # Calculate the global max value from all frames
    global_max_val = np.max(values)

    plt.style.use(['science', 'nature', 'no-latex'])
    # Create a figure for the visualization
    fig, ax = plt.subplots(figsize=(3, 2))

    # Initialize scatter plot with dot size
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values[0], cmap='twilight_shifted', vmin=0, vmax=global_max_val, s=0.4)
    cbar = fig.colorbar(scatter, ax=ax, label='Koncentracija')
    #cbar = fig.colorbar(scatter, ax=ax, label='Concentration')

    ax.set_title('t = {:.0f} s'.format(time_values[0]))
    ax.set_xlabel('X / $\mathrm{μm}$')
    ax.set_ylabel('Y / $\mathrm{μm}$')
    ax.set_xlim([coords[:, 0].min(), coords[:, 0].max()])
    ax.set_ylim([coords[:, 1].min(), coords[:, 1].max()])

    # Iterate over the found frames and save the corresponding images
    for i, frame in enumerate(frames_to_save):
        temp_values = values[frame]
        scatter.set_array(temp_values)
        scatter.set_clim(vmin=0, vmax=global_max_val)  # Fix color scale based on global max value
        ax.set_title('t = {:.1f} s'.format(times_of_interest[i]))
        
        # Save the current figure as an image
        plt.savefig(f'value_scatter_time_{label}_{time_values[frame]:.0f}.png', dpi=150)

    # Display the final plot (optional, can be removed if only saving images)
    # plt.show()



import json

def load_json_to_tuples(file_path):
    """
    Reads JSON data from a file and converts it to a list of tuples.
    
    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    list of tuples: Converted data.
    """
    with open(file_path, 'r') as file:
        data_list = json.load(file)
    
    # Convert list of lists to list of tuples
    tuple_list = [tuple(item) for item in data_list]
    
    return tuple_list

# Paths to the JSON files
#neurons_file = 'nerves_coordinates.json'
#capillaries_file = 'capillaries_coordinates.json'



#plot two over half curves:

#file_path2 = 'hex_grid_results_20_20_bupi_acid_in_ok.csv'
#data2 = pd.read_csv(file_path2)

def plot_in(data1, data2, label):
    num_neurons = len(neurons)

    c0 = 1.

    # Function to calculate the ratio of neurons with concentration over 0.5
    def calculate_proc_over_half(dataset):
        proc_over_half = []
        for i in range(len(dataset.iloc[:, 0])):
            a = 0
            for j in range(num_neurons):
                a += 1 if dataset.iloc[i, j] >= (c0/2) else 0
            a = a / num_neurons
            proc_over_half.append(a)
        return proc_over_half

    # Function to calculate the ratio of neurons with concentration over 0.5
    def calculate_proc_over_half_2(times, values):
        proc_over_half = []
        for i in range(len(times)):
            a = 0
            for j in range(num_neurons):
                a += 1 if values[i, j] >= (c0/2) else 0
            a = a / num_neurons
            proc_over_half.append(a)
        return proc_over_half

    def calculate_avg_conc(dataset):
        avg_c = []
        for i in range(len(dataset.iloc[:, 0])):
            a = 0
            for j in range(num_neurons):
                a += dataset.iloc[i, j]
            a = a / num_neurons
            avg_c.append(a)
        return avg_c
    
    def get_times(data):
        columns = data.columns[1:]
        proc_columns = []
        for col in columns:
            str_col = col[2:-1]
            spl_col = str_col.split(', ')
            flt_col = [float(i) for i in spl_col]
            proc_columns.append(flt_col)

        coords = np.array(proc_columns)
        values = np.array(data.iloc[:, 1:].values)
        times = np.array(data.iloc[:, 0].values)
        return times

    times = get_times(data1)
    # Calculate for both data sets
    proc_over_half_data1 = calculate_proc_over_half(data1)
    proc_over_half_data2 = calculate_proc_over_half(data2)
    avg_c_data1 = calculate_avg_conc(data1)
    avg_c_data2 = calculate_avg_conc(data2)

    # Find tau_i for data1
    tau_i_data1 = -1
    for i in range(len(proc_over_half_data1)):
        if proc_over_half_data1[i] >= 0.5:
            tau_i_data1 = i
            break
    print(tau_i_data1)
    print(data1.iloc[tau_i_data1, 0])
    print(times[-1])

    # Find tau_i for data2
    tau_i_data2 = -1
    for i in range(len(proc_over_half_data2)):
        if proc_over_half_data2[i] >= 0.5:
            tau_i_data2 = i
            break
    print(tau_i_data2)
    print(data2.iloc[tau_i_data2, 0])

    # Plotting both curves
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.figure(figsize=(4, 3))
    plt.plot(times, proc_over_half_data1, label='Fizioloski')
    plt.plot(data2.iloc[:, 0], proc_over_half_data2, label='Acidoza', linestyle='--')
    plt.xlabel('t / s')
    plt.ylabel('Delez aksonov s koncentracijo nad 0.5')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'part_over_fifty_{label}.png', dpi=300)
    #plt.show()

    plt.figure(figsize=(4, 3))
    plt.plot(data1.iloc[:, 0], avg_c_data1, label='Fizioloski')
    plt.plot(data2.iloc[:, 0], avg_c_data2, label='Acidoza', linestyle='--')
    plt.xlabel('t / s')
    plt.ylabel('Povprecna koncentracija')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'acerage_c_{label}.png', dpi=300)
    #plt.show()



from scipy.optimize import curve_fit


def plot_out(data1, data2, label):
    num_neurons = len(neurons)

    c0 = 1.
    c0_1  = max(data1.iloc[0, :])
    c0_2  = max(data2.iloc[0, :])

    # Function to calculate the ratio of neurons with concentration over 0.5
    def calculate_proc_over_half(dataset):
        proc_over_half = []
        for i in range(len(dataset.iloc[:, 0])):
            a = 0
            for j in range(num_neurons):
                a += 1 if dataset.iloc[i, j] >= (c0/2) else 0
            a = a / num_neurons
            proc_over_half.append(a)
        return proc_over_half

    def calculate_avg_conc(dataset):
        avg_c = []
        for i in range(len(dataset.iloc[:, 0])):
            a = 0
            for j in range(num_neurons):
                a += dataset.iloc[i, j]
            a = a / num_neurons
            avg_c.append(a)
        return avg_c

    # Calculate for both data sets
    proc_over_half_data1 = calculate_proc_over_half(data1)
    proc_over_half_data2 = calculate_proc_over_half(data2)
    avg_c_data1 = calculate_avg_conc(data1)
    avg_c_data2 = calculate_avg_conc(data2)

    # Find tau_i for data1
    tau_i_data1 = -1
    for i in range(len(proc_over_half_data1)):
        if proc_over_half_data1[i] >= 0.5:
            tau_i_data1 = i
            break
    print(tau_i_data1)
    print(data1.iloc[tau_i_data1, 0])

    # Find tau_i for data2
    tau_i_data2 = -1
    for i in range(len(proc_over_half_data2)):
        if proc_over_half_data2[i] >= 0.5:
            tau_i_data2 = i
            break
    print(tau_i_data2)
    print(data2.iloc[tau_i_data2, 0])



    # Define the exponential decay function with an asymptote
    def exp_decreasing1(x, B):
        #return c0_1 - c0_1 * np.exp(-B * x)
        return 0 + c0_1 * np.exp(-B * x)
    def exp_decreasing2(x, B):
        #return c0_2 - c0_2 * np.exp(-B * x)
        return 0 + c0_2 * np.exp(-B * x)

    # Example data
    x_data = np.array(data1.iloc[:, 0]*3.18)  # Replace with the x values from your data

    # Initial guess for the parameter: [B]
    initial_guess = [0.1]  # Initial guess for B

    # Fit the exponential decay model
    try:
        popt1, _ = curve_fit(exp_decreasing1, x_data, avg_c_data1, p0=initial_guess)
        B_fit1 = popt1[0]
        print(f'Fitted parameters:\nB = {B_fit1:.3g}')
        popt2, _ = curve_fit(exp_decreasing2, x_data, avg_c_data2, p0=initial_guess)
        B_fit2 = popt2[0]
        print(f'Fitted parameters:\nB = {B_fit2:.3g}')
    except Exception as e:
        print(f"An error occurred: {e}")
        B_fit1 = np.nan
        B_fit2 = np.nan

    plt.style.use(['science', 'nature', "no-latex"])
    # Plot the data and the fitted curve
    plt.figure(figsize=(4, 3), dpi=300)
    #plt.plot(x_data, avg_conc, 'o', label='Data')
    x_fit = np.linspace(0,10000, 1000)  # Extend x values up to 10
    y_fit1 = exp_decreasing1(x_fit, B_fit1)
    y_fit2 = exp_decreasing2(x_fit, B_fit2)
    plt.plot(x_fit, y_fit1, '-', label='Fizioloski')
    plt.plot(x_fit, y_fit2, '--', label='Acidoza')
    plt.xlabel('t / s')
    plt.ylabel('Povprecna koncentracija')
    #plt.title('Exponential Decay Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'avg_conc_fit_{label}.png', dpi=300)
    #plt.show()



    plt.figure(figsize=(4, 3), dpi=300)
    x_fit = np.linspace(0,10000, 1000)  # Extend x values up to 10
    # Function to calculate the ratio of neurons with concentration over 0.5
    def calculate_proc_over_half_2(times, values, c0):
        proc_over_half = []
        for i in range(len(times)):
            a = 0
            if values[i] >= (c0/2):
                a = 1
            proc_over_half.append(a)
        for i in range(10):
            temp_proc = proc_over_half
            for j in range(1,len(times)-1):
                proc_over_half[j] = (temp_proc[j-1] + temp_proc[j] + temp_proc[j+1])/3
        return proc_over_half
    print(c0_1, c0_2)
    proc_over_half_data1 = calculate_proc_over_half_2(x_fit, y_fit1, c0_1)
    proc_over_half_data2 = calculate_proc_over_half_2(x_fit, y_fit2, c0_2)


    plt.plot(x_fit, proc_over_half_data1, '-', label='Fizioloski')
    plt.plot(x_fit, proc_over_half_data2, '--', label='Acidoza')
    plt.xlabel('t / s')
    plt.ylabel('Delez aksonov s koncentracijo nad 0.5')
    #plt.title('Exponential Decay Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'overfifty_fit_{label}.png', dpi=300)
    #plt.show()


neurons_file = 'neurons_20_hex.json'
capillaries_file = 'capillaries_20_hex.json'
# Load data from each file
neurons = load_json_to_tuples(neurons_file)
capillaries = load_json_to_tuples(capillaries_file)
hex_grid, border_points = generate_grid_from_real_points(neurons, capillaries, 3.5)
#hex_grid, border_points = generate_grid_for_void(neurons, capillaries, 3.5)
plot_hex_grid(hex_grid)

"""
coords, values, times = import_results("hex_grid_20_10_lido_in.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 1.,5.,10.], coords, values, times, "lido_in")
coords, values, times = import_results("hex_grid_results_20_20_lido_acid_in_ok.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 1.,5.,10.], coords, values, times, "lido_acid_in")
coords, values, times = import_results("hex_grid_results_20_20_bupi_in_ok.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 2.,10.,20.], coords, values, times, "bupi_in")
coords, values, times = import_results("hex_grid_results_20_20_bupi_acid_in_ok.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 2.,10.,20.], coords, values, times, "bupi_acid_in")
coords, values, times = import_results("hex_grid_20_100_lido_out.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 10.,50.,100.], coords, values, times, "lido_out")
coords, values, times = import_results("hex_grid_20_100_lido_acid_out.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 10.,50.,100.], coords, values, times, "lido_acid_out")
coords, values, times = import_results("hex_grid_20_1000_bupi_out.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 100.,500.,1000.], coords, values, times, "bupi_out")
coords, values, times = import_results("hex_grid_20_1000_bupi_acid_out.csv")
coords, values = mirror(coords, values)
plot_time_series([0., 100.,500.,1000.], coords, values, times, "bupi_acid_out")
"""
"""
neurons_file = 'neurons_20.json'
# Load data from each file
neurons = load_json_to_tuples(neurons_file)

data1 = pd.read_csv("hex_grid_20_10_lido_in.csv")
data2 = pd.read_csv("hex_grid_results_20_10_lido_acid_in_ok.csv")
plot_in(data1, data2, "lido")
data1 = pd.read_csv("hex_grid_results_20_20_bupi_in_ok.csv")
data2 = pd.read_csv("hex_grid_results_20_20_bupi_acid_in_ok.csv")
plot_in(data1, data2, "bupi")


data1 = pd.read_csv("hex_grid_20_100_lido_out.csv")
data2 = pd.read_csv("hex_grid_20_100_lido_acid_out.csv")
plot_out(data1, data2, "lido")
data1 = pd.read_csv("hex_grid_20_1000_bupi_out.csv")
data2 = pd.read_csv("hex_grid_20_1000_bupi_acid_out.csv")
plot_out(data1, data2, "bupi")"""