# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import csv
import sys

fill_or_void = 0 #0=fill, 1=void
lido = False
bupi = True
acid = False
time = 5 #s
num_output_points = 100
real_nerve = False
starting_c_file = "hex_grid_20_5_bupi_in_ok.csv"

# %%

def generate_hexagonal_grid(n_layers, neuron_r, ecf_proc):
    # Constants for hexagon geometry
    cos_60 = 0.5
    sin_60 = math.sqrt(3) / 2
    directions = [(2, 0), (1, 2), (1, -2), (-2, 0), (-1, 2), (-1, -2)]
    
    r_ecf = np.sqrt(ecf_proc/(1-ecf_proc) * neuron_r**2 / 3)
    
    distance_neurons = 2. * neuron_r + 2. * r_ecf
    x_math_correction = 0.5
    y_math_correction = 0.5 * sin_60
    
    print(distance_neurons)

    def hex_neighbours(x, y):
        return [(x + dx, y + dy) for dx, dy in directions]

    points = set()
    central_point = (0, 0)
    points.add(central_point)
    
    last_points = [central_point]
    for layer in range(n_layers):
        new_points = []
        for (x, y) in last_points:
            neighbours = hex_neighbours(x, y)
            for point in neighbours:
                if point not in points:
                    new_points.append(point)
        last_points = new_points
        for point in new_points:
            points.add(point)

    points_list = list(points)
    
    corrected_points_list= [(x * distance_neurons * x_math_correction, y * distance_neurons * y_math_correction) for x, y in points_list]

    return corrected_points_list


# %%
def generate_hexagonal_triangle(n_layers, neuron_r, ecf_proc):
    # Constants for hexagon geometry
    cos_60 = 0.5
    sin_60 = math.sqrt(3) / 2
    directions = [(2, 0), (1, 2), (1, -2), (-2, 0), (-1, 2), (-1, -2)]
    
    r_ecf = np.sqrt(ecf_proc/(1-ecf_proc) * neuron_r**2 / 3)
    
    distance_neurons = 2. * neuron_r + 2. * r_ecf
    x_math_correction = 0.5
    y_math_correction = 0.5 * sin_60

    # Function to check if a point is within the triangle
    def in_triangle(x, y):
        return (0 <= x <= y/math.sqrt(3)*1.02) and (x*math.sqrt(3)*0.98 <= y)

    def hex_neighbours(x, y):
        return [(x + dx, y + dy) for dx, dy in directions]

    points = set()
    central_point = (0, 0)
    points.add(central_point)
    
    last_points = [central_point]
    for layer in range(n_layers):
        new_points = []
        for (x, y) in last_points:
            neighbours = hex_neighbours(x, y)
            for point in neighbours:
                if point not in points:
                    new_points.append(point)
        last_points = new_points
        for point in new_points:
            points.add(point)

    points_list = list(points)
    
    # Adjust points to be centered at (0,0)
    corrected_points_list= [(x * distance_neurons * x_math_correction, y * distance_neurons * y_math_correction) for x, y in points_list]
    
    triangle_points = []
    for point in corrected_points_list:
        if in_triangle(*point):
            triangle_points.append(point)
        

    return triangle_points

# %%
import math
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

# %%
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from shapely.geometry import LineString

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

def plot_results(original_points, convex_hull_points, offset_polygon, border_points):
    """
    Plots the original points, convex hull, offset polygon, and border points.

    Parameters:
    original_points (list of tuples): List of original points.
    convex_hull_points (list of tuples): List of convex hull points.
    offset_polygon (shapely.geometry.Polygon): The offset polygon.
    border_points (list of tuples): List of border points.
    """
    # Convert lists to numpy arrays for easier plotting
    original_points_array = np.array(original_points)
    convex_hull_points_array = np.array(convex_hull_points)
    border_points_array = np.array(border_points)
    offset_polygon_coords = np.array(offset_polygon.exterior.coords)
    
    plt.figure(figsize=(10, 8))
    
    # Plot original points
    plt.scatter(*original_points_array.T, color='blue', label='Original Points', zorder=5)
    
    # Plot convex hull
    plt.plot(*np.append(convex_hull_points_array, [convex_hull_points_array[0]], axis=0).T, color='green', label='Convex Hull', linestyle='--', zorder=4)
    
    # Plot offset polygon
    plt.plot(*np.append(offset_polygon_coords, [offset_polygon_coords[0]], axis=0).T, color='orange', label='Offset Polygon', linestyle='-', zorder=3)
    
    # Plot border points
    plt.scatter(*border_points_array.T, color='red', label='Border Points', zorder=6)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Convex Hull, Offset Polygon, and Border Points')
    plt.legend()
    plt.grid(True)
    plt.show()



# %%
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
    plot_results(input_points, hull_points, offseted_polygon, raw_border_points)
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

# %%
points= [
    (0,0),
    (1,0),
    (0,2),
    (-1,0)
]

grid, border_points = generate_grid_from_real_points(points, [(0,4)], 1)

print(grid)

# %%
def generate_grid_for_void(input_points_main, input_points_aux2, max_distance):
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
                #grid[i][3].append(np_index)
                #grid[j][3].append(np_index)
    for i in range(len(input_points_main)):
        for j in range(len(input_points_aux2)):
            np_index = len(input_points_main) + j
            grid[i][3].append(np_index)
            grid[np_index][3].append(i)
    # Generate convex hull
    hull_points = generate_convex_hull(input_points)
    hull_polygon = Polygon(hull_points)

    # Offset polygon and generate border points
    border_offset = 1
    border_spacing = 3

    offseted_polygon = offset_polygon(hull_polygon, border_offset)
    raw_border_points = generate_border_points(offseted_polygon, border_spacing)
    plot_results(input_points, hull_points, offseted_polygon, raw_border_points)
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
                #grid[j][3].append(np_index)
                #border_points[i][3].append(j)
                pass
        for j in range(num_new_points):
            dist = calculate_distance(new_points[j], border_points[i])
            if dist <= max_distance:
                #new_points[j][3].append(np_index)
                #border_points[i][3].append(num_points + j)
                pass
        
    print(len(grid))
    grid.extend(new_points)
    print(len(grid))
    grid.extend(border_points)
    print(len(grid))
    return grid, border_points

# %%

def plot_hex_grid(hex_grid):
    main_points = [point for point in hex_grid if point[2] == 'main']
    aux_points = [point for point in hex_grid if point[2] == 'aux']
    aux2_points = [point for point in hex_grid if point[2] == 'aux2']
    
    print(len(main_points))
    print(len(aux_points))
    print(len(aux2_points))
    
    plt.figure(figsize=(8,8), dpi = 150)
    
    # Plot aux points
    if aux_points:
        aux_x, aux_y = zip(*[(x, y) for x, y, _, _ in aux_points])
        plt.scatter(aux_x, aux_y, color='green', label='ECF')
    
    # Plot main points
    if main_points:
        main_x, main_y = zip(*[(x, y) for x, y, _, _ in main_points])
        plt.scatter(main_x, main_y, color='blue', label='NEre Fiber')
        
    if aux2_points:
        main_x, main_y = zip(*[(x, y) for x, y, _, _ in aux2_points])
        plt.scatter(main_x, main_y, color='red', label='Capillary')
    
    # Label each point with its index
    for i, (x, y, _, _) in enumerate(hex_grid):
        plt.text(x, y, str(i), fontsize=9, ha='right')
    
    #plt.title('Hexagonal Grid with Main and Aux Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# %%
def create_ode_functions(hex_grid, volumes, thickness, kaux, kam, kma, kcap, D_coeff):
    #volumes in yl, coonvert them to ym^3
    ym3volumes = [i * 10e9 for i in volumes]
    def create_ode_function(index, point, neighbours, k_intrinsic, k_matrix, volumes, thickness):
        def ode(C, t):
            C_i = C[index]
            sum_k_ji_Cj = 0
            sum_k_ij_Ci = 0
            for j in neighbours:
                k_ij = k_matrix[index][j][0]
                k_ji = k_matrix[index][j][1]
                sum_k_ij_Ci += k_ij * C_i
                sum_k_ji_Cj += k_ji * C[j]
            return -k_intrinsic[index] * C_i + sum_k_ji_Cj - sum_k_ij_Ci
        return ode

    n_points = len(hex_grid)
    k_matrix = np.zeros((n_points, n_points, 2))
    k_intrinsic = np.zeros(n_points)

    for i, (x, y, point_type, neighbours) in enumerate(hex_grid):
        for j in neighbours:
            _, _, neighbour_type, _ = hex_grid[j]
            if (point_type == 'aux' or point_type == 'aux2') and (neighbour_type == 'aux' or neighbour_type == 'aux2'):
                k_matrix[i][j][0] = 2 * D_coeff / (math.sqrt(ym3volumes[i]/(np.pi*thickness))+math.sqrt(ym3volumes[j]/(np.pi*thickness)))**2
                k_matrix[i][j][1] = 2 * D_coeff / (math.sqrt(ym3volumes[i]/(np.pi*thickness))+math.sqrt(ym3volumes[j]/(np.pi*thickness)))**2
            elif (point_type == 'aux' or point_type == 'aux2') and neighbour_type == 'main':
                k_matrix[i][j][0] = (2 * D_coeff * kam / volumes[j]) / (ym3volumes[i]/(np.pi*thickness) * kam / volumes[j] + 2 * D_coeff + ym3volumes[j]/(np.pi*thickness) * kam / volumes[j]) * volumes[j] / volumes[i]
                k_matrix[i][j][1] = (2 * D_coeff * kam / volumes[j]) / (ym3volumes[i]/(np.pi*thickness) * kam / volumes[j] + 2 * D_coeff + ym3volumes[j]/(np.pi*thickness) * kam / volumes[j]) * volumes[j] / volumes[i] * kma/kam
                pass
            elif point_type == 'main' and (neighbour_type == 'aux' or neighbour_type == 'aux2'):
                k_matrix[i][j][0] = (2 * D_coeff * kam / volumes[i]) / (ym3volumes[i]/(np.pi*thickness) * kam / volumes[i] + 2 * D_coeff + ym3volumes[j]/(np.pi*thickness) * kam / volumes[i]) * kma/kam
                k_matrix[i][j][1] = (2 * D_coeff * kam / volumes[i]) / (ym3volumes[i]/(np.pi*thickness) * kam / volumes[i] + 2 * D_coeff + ym3volumes[j]/(np.pi*thickness) * kam / volumes[i])
        
        if point_type == 'aux2':
            k_intrinsic[i] = kcap / volumes[i]
    
    
    ode_functions = []
    for i, (x, y, _, neighbours) in enumerate(hex_grid):
        ode_func = create_ode_function(i, (x, y), neighbours, k_intrinsic, k_matrix, volumes, thickness)
        ode_functions.append(ode_func)
    
    return ode_functions

# %%
def static_ode_function():
    def ode(C, t):
        return 0
    return ode

# %%

# ODE system with dynamic output
def system_of_odes(t, C, ode_functions):
    dCdt = np.zeros(len(C))
    
    for i, ode in enumerate(ode_functions):
        dCdt[i] = ode(C, t)
    print(f"t = {t:.5f}")   
    sys.stdout.flush() 
    return dCdt


# %%
def calculate_volumes(hex_grid, length, main_radius):
    """
    Calculate the volume of a sphere for points of type 'aux' and 'aux2' with a radius
    that is half the average distance between each point and its neighbors.
    Points of type 'main' will have a hardcoded volume value.

    Parameters:
    hex_grid (list of tuples): List of points in the format (x, y, point_type, neighbours),
                               where neighbours are indices of other points in the list.
    main_volume (float): Hardcoded volume for points of type 'main'.

    Returns:
    list of floats: Volumes of the spheres or hardcoded values for each point.
    """
    
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def average_half_distance_from_neighbors(index, hex_grid):
        point = hex_grid[index]
        x, y, _, neighbors = point
        distances = []
        for neighbor_index in neighbors:
            if 0 <= neighbor_index < len(hex_grid):
                neighbor = hex_grid[neighbor_index]
                nx, ny, _, _ = neighbor
                dist = euclidean_distance((x, y), (nx, ny))
                distances.append(dist)
        return np.mean(distances) / 2 if distances else 0

    def volume_of_cylinder(radius, length):
        return np.pi * radius**2 * length

    volumes = []
    for i in range(len(hex_grid)):
        point = hex_grid[i]
        _, _, point_type, _ = point
        
        if point_type == 'main':
            volume = volume_of_cylinder(main_radius, length)
            volumes.append(volume)
        else:
            avg_distance = average_half_distance_from_neighbors(i, hex_grid)
            volume = volume_of_cylinder(avg_distance-main_radius, length)
            volumes.append(volume)

    return volumes


# %%

def save_results_to_csv(solution, hex_grid, filename):
    columns = ['Time'] + [f'C({x}, {y})' for x, y, _, _ in hex_grid]
    data = np.column_stack((solution.t, solution.y.T))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

def initialize_csv(hex_grid, filename):
    columns = ['Time'] + [f'C({x}, {y})' for x, y, _, _ in hex_grid]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
    return filename

# %%
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

if real_nerve:
# Paths to the JSON files
    neurons_file = 'nerves_coordinates.json'
    capillaries_file = 'capillaries_coordinates.json'
else:
    neurons_file = 'neurons_20.json'
    capillaries_file = 'capillaries_20.json'

# Load data from each file
neurons = load_json_to_tuples(neurons_file)
capillaries = load_json_to_tuples(capillaries_file)

# Print the results
print("Neurons Data:", neurons)
print("Capillaries Data:", capillaries)


# %%
# Example usage
#n_layers = 2
#neurons = generate_hexagonal_grid(n_layers, 1., 0.6)
#capillaries = []

# %%
# Example usage
#n_layers = 30
#neurons = generate_hexagonal_triangle(n_layers, 1., 0.6)
#capillaries = []

# %%
#print(len(neurons))
#ratio = 0.0026

#import random

#for r in range(int(math.ceil(len(neurons)*ratio))):
#    i = random.randint(0,len(neurons))
#    temp_element = neurons.pop(i)
#    capillaries.append(temp_element)



# %%
#import json
#with open("neurons.json", 'w') as f:
#    json.dump(neurons, f)

#with open("capillaries.json", 'w') as f:
#    json.dump(capillaries, f)

# %%
if real_nerve:
    modifier = 0.05886
    neurons = [(x * modifier, y * modifier) for x,y in neurons]
    capillaries = [(x * modifier, y * modifier) for x,y in capillaries]

# %%
if fill_or_void == 0:
    if real_nerve:
        hex_grid, border_points = generate_grid_from_real_points(neurons, capillaries, 20)
    else:
        hex_grid, border_points = generate_grid_from_real_points(neurons, capillaries, 3.5)
elif fill_or_void == 1:
    if real_nerve:
        hex_grid, border_points = generate_grid_for_void(neurons, capillaries, 20)
    else:
        hex_grid, border_points = generate_grid_for_void(neurons, capillaries, 3.5)


plot_hex_grid(hex_grid)

segment_length = 1500
fib_rad = 1

volumes = calculate_volumes(hex_grid, segment_length, fib_rad)

volumes = [i * 10e-9 for i in volumes] #convert from 10e-18 m3 to 10e-9 m3 (yl)

print(volumes)

# Example intrinsic k values for each point in unit yl s-1
k_intrinsic = [0.0 for _ in range(len(hex_grid))]
kaa = 10
if lido and not acid:
    D=749 # in units ym^2 s^-2 for lidocaine
    kam = 9.147e-3
    kma = 2.152e-4
elif lido and acid:
    D=749 # in units ym^2 s^-2 for lidocaine
    kam = 2.322e-2
    kma = 1.942e-3
elif bupi and not acid:
    D=671 # in units ym^2 s^-2 for lidocaine
    kam = 3.774e-3
    kma = 1.093e-5
elif bupi and acid:
    D=671 # in units ym^2 s^-2 for lidocaine
    kam = 6.948e-3
    kma = 7.127e-5
kcap = -5.09e-8

# %%
ode_functions = create_ode_functions(hex_grid, volumes, segment_length, kaa, kam, kma, kcap, D)

# Initial concentrations
C0 = np.zeros(len(hex_grid))

if starting_c_file != "":
    print("using starting file")
    # Load the CSV file
    data = pd.read_csv(starting_c_file)

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
    for i, a in enumerate(coords):
        for j, b in enumerate(hex_grid):
            if a[0] == b[0] and a[1] == b[1]:
                C0[j] = values[-1, i]
    print(C0)

if fill_or_void == 0:
    max_y = max(i[1] for i in hex_grid)
    if real_nerve:
        indices = [i for i, x in enumerate(hex_grid) if x in border_points]
    else:
        indices = [i for i, x in enumerate(hex_grid) if x[1]==max_y]
    for i in indices:
        C0[i] = 1
    for i in indices:
        ode_functions[i] = static_ode_function()
elif fill_or_void == 1:
    for i, x in enumerate(hex_grid):
        if x[2] == "main":
            C0[i] = kam/kma
        elif x[2] == "aux":
            C0[i] = 1
        elif x[2] == "aux2":
            C0[i] = 0
            ode_functions[i] = static_ode_function()

                

# Time span for the integration
t_span = (0, time)
# Time points where the solution is computed
t_eval = np.linspace(0, time, num_output_points)

file_name = "hex_grid"
if real_nerve:
    file_name += "_real"
else:
    file_name += "_20"
file_name += "_" + str(time)
if lido:
    file_name += "_lido"
elif bupi:
    file_name += "_bupi"
if acid:
    file_name += "_acid"
if fill_or_void == 0:
    file_name += "_in"
else:
    file_name += "_out"
file_name += ".csv"

sys.stdout.flush()
solution = solve_ivp(system_of_odes, t_span, C0, t_eval=t_eval, args=(ode_functions,), method='RK45', max_step=0.001)


# Save the results to a CSV file
save_results_to_csv(solution, hex_grid, file_name)

# %%
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_networkx_graph_from_hex_grid(hex_grid):
    """
    Create a NetworkX graph from the hex grid data.

    Parameters:
    - hex_grid (list of tuples): List of tuples (x, y, type, neighbours).

    Returns:
    - G (networkx.Graph): The generated graph.
    """
    G = nx.Graph()
    
    # Define node colors based on type using a colormap
    color_map = {
        'main': 'red',
        'aux': 'green',
        'aux2': 'blue'
    }
    
    # Add nodes with positions and types
    for index, (x, y, type_, _) in enumerate(hex_grid):
        G.add_node(index, pos=(x, y), type=type_)
    
    # Add edges based on neighbors
    for index, (_, _, _, neighbors) in enumerate(hex_grid):
        for neighbor_index in neighbors:
            if not G.has_edge(index, neighbor_index):
                G.add_edge(index, neighbor_index)
    
    return G

def visualize_graph(G):
    """
    Visualize the NetworkX graph with node color-coding based on type and add a legend.

    Parameters:
    - G (networkx.Graph): The graph to visualize.
    
    Returns:
    - None
    """
    pos = nx.get_node_attributes(G, 'pos')
    types = nx.get_node_attributes(G, 'type')
    
    # Define a colormap
    color_map = {
        'main': plt.get_cmap('Blues')(0.7),
        'aux': plt.get_cmap('Greens')(0.7),
        'aux2': plt.get_cmap('Reds')(0.7)
    }
    
    node_colors = [color_map.get(types[node], 'gray') for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_weight='bold')
    
    # Create a legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Živčno vlakno', markersize=10, markerfacecolor=color_map['main']),
        plt.Line2D([0], [0], marker='o', color='w', label='ZuCT', markersize=10, markerfacecolor=color_map['aux']),
        plt.Line2D([0], [0], marker='o', color='w', label='Kapilara', markersize=10, markerfacecolor=color_map['aux2'])
    ]
    
    plt.legend(handles=handles)
    plt.show()


#n_layers = 1
#hex_grid = generate_hexagonal_grid(n_layers)

hex_grid = [
    (0.0, 0.0, 'main', [1, 2]),   # Node 0, type 'main', neighbors are nodes 1 and 2
    (1.0, 0.0, 'aux', [0, 2]), # Node 1, type 'aux', neighbors are nodes 0, 2, and 3
    (0.5, 0.86, 'aux2', [0, 1]),   # Node 2, type 'aux2', neighbors are nodes 0 and 1
]

G = create_networkx_graph_from_hex_grid(hex_grid)
visualize_graph(G)



