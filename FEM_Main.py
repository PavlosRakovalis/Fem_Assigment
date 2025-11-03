import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from geometry_utils import rotate_points_around_y
import plotly.graph_objects as go
import plotly.io as pio

# Set browser for Plotly plots
# Options: 'chrome', 'firefox', 'edge', or None for default browser
BROWSER = 'chrome'  # Change to 'firefox', 'edge', or None

if BROWSER:
    pio.renderers.default = 'browser'


# Function to set 3D axes to equal scale for accurate geometry perception of the plotings. 
# We are using it in the matplotlib plots below.  
def set_3d_axes_equal(ax, xs, ys, zs):
    """Make 3D axes have equal scale so that spheres appear as spheres.

    This sets xlim, ylim, zlim to the same half-range around each axis midpoint,
    and (if available) enforces a unit box aspect.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)

    x_mid = (xs.max() + xs.min()) / 2.0
    y_mid = (ys.max() + ys.min()) / 2.0
    z_mid = (zs.max() + zs.min()) / 2.0

    max_range = 0.5 * max(np.ptp(xs), np.ptp(ys), np.ptp(zs), 1e-9)

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # Matplotlib 3.3+ supports set_box_aspect
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def display_matrix_table(df, title="Stiffness Matrix"):
    """
    Display a pandas DataFrame in an interactive, scrollable table window.
    """
    root = tk.Tk()
    root.title(title)
    root.geometry("1000x600")
    
    # Create frame for the table with scrollbars
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create Treeview widget (table)
    tree = ttk.Treeview(frame, show='tree headings')
    
    # Create scrollbars
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Grid layout for table and scrollbars
    tree.grid(column=0, row=0, sticky='nsew')
    vsb.grid(column=1, row=0, sticky='ns')
    hsb.grid(column=0, row=1, sticky='ew')
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    # Define columns (first column is row labels)
    tree["columns"] = ["Row"] + list(df.columns)
    
    # Format column headers
    tree.column("#0", width=0, stretch=tk.NO)  # Hide the default first column
    tree.heading("#0", text="", anchor=tk.W)
    
    tree.column("Row", anchor=tk.W, width=80)
    tree.heading("Row", text="", anchor=tk.W)
    
    for col in df.columns:
        tree.column(col, anchor=tk.CENTER, width=80)
        tree.heading(col, text=col, anchor=tk.CENTER)
    
    # Insert data rows
    for idx, row in df.iterrows():
        values = [idx] + [f"{val:.6f}" if isinstance(val, (int, float)) else str(val) 
                         for val in row]
        tree.insert("", tk.END, values=values)
    
    # Add status bar
    status_bar = ttk.Label(root, text=f"Matrix size: {df.shape[0]} × {df.shape[1]}", 
                          relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    root.mainloop()






























####################################   DEFINING NODES   ######################




# Creating a DataFrame with node numbers and coordinates
#it is empty for now
points = pd.DataFrame({
    'Node Number': [],
    'X': [],
    'Y': [],
    'Z': []
})



L = 1.5 * 1.13
A = 1.2 * 1.69
phi = 60 + 9.1 #degrees
A_0 = 6 * (0.5 + 0.13 )


# Arhika estw oti to simeio 0 ,0,0 einai stin thesi opou pianei o geranos 



# Add a new point at x = A, y = -L/2, z = 0
new_node = pd.DataFrame({
    'Node Number': [len(points) + 1],
    'X': [A],
    'Y': [-L/2],
    'Z': [0.0]
})
points = pd.concat([points, new_node], ignore_index=True)

# Add a new point at x = A, y = +L/2, z = 0
new_node = pd.DataFrame({
    'Node Number': [len(points) + 1],
    'X': [A],
    'Y': [L/2],
    'Z': [0.0]
})
points = pd.concat([points, new_node], ignore_index=True)



# Add 7 points at z = -L/2, y = -L/2, spaced by L in x dimension starting at x = A+L
for i in range(7):
    new_node = pd.DataFrame({
        'Node Number': [len(points) + 1],
        'X': [A + L + i * L],
        'Y': [-L/2],
        'Z': [-L/2]
    })
    points = pd.concat([points, new_node], ignore_index=True)



# Add 7 points at z = -L/2, y = L/2, spaced by L in x dimension starting at x = A+L
for i in range(7):
    new_node = pd.DataFrame({
        'Node Number': [len(points) + 1],
        'X': [A + L + i * L],
        'Y': [L/2],
        'Z': [-L/2]
    })
    points = pd.concat([points, new_node], ignore_index=True)


# Add 6 points at z = +L/2, y = -L/2, spaced by L in x dimension starting at x = A+L
for i in range(6):
    new_node = pd.DataFrame({
        'Node Number': [len(points) + 1],
        'X': [A + L + i * L],
        'Y': [-L/2],
        'Z': [L/2]
    })
    points = pd.concat([points, new_node], ignore_index=True)


# Add 6 points at z = L/2, y = L/2, spaced by L in x dimension starting at x = A+L
for i in range(6):
    new_node = pd.DataFrame({
        'Node Number': [len(points) + 1],
        'X': [A + L + i * L],
        'Y': [L/2],
        'Z': [L/2]
    })
    points = pd.concat([points, new_node], ignore_index=True)

# Add a point at x = A + L * 6.5, y = 0, z = -1.5*L
new_node = pd.DataFrame({
    'Node Number': [len(points) + 1],
    'X': [A + L * 6.5],
    'Y': [0],
    'Z': [-1.5 * L]
})
points = pd.concat([points, new_node], ignore_index=True)






























###############################   DEFINING ELEMENTS   ######################



#### Define elements connecting the points and plot them



elements = pd.DataFrame({
    'Element Number': [],
    'Node1': [],
    'Node2': [],
    'E': [],
    'V': []
})





# Automatically create elements by connecting neighboring points
# Only create elements parallel to X, Y, or Z axes

elements = pd.DataFrame(columns=['Element Number', 'Node1', 'Node2', 'E', 'V'])
element_counter = 1
tolerance = 1e-6  # Tolerance for coordinate comparison
max_element_length = 1.55 * L  # Maximum allowed element length

# For each point, find neighbors along X, Y, or Z axes
for i in range(len(points)):
    node1 = points.iloc[i]
    node1_num = node1['Node Number']
    
    # Check all other points
    for j in range(i + 1, len(points)):
        node2 = points.iloc[j]
        node2_num = node2['Node Number']
        
        # Calculate differences
        dx = abs(node2['X'] - node1['X'])
        dy = abs(node2['Y'] - node1['Y'])
        dz = abs(node2['Z'] - node1['Z'])
        
        # Calculate element length
        element_length = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Skip if element is too long
        if element_length > max_element_length:
            continue
        
        # Check if element is parallel to X-axis (only X changes)
        if dx > tolerance and dy < tolerance and dz < tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],  # Default Young's modulus (Pa)
                'V': [0.3]     # Default Poisson's ratio
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            element_counter += 1
        
        # Check if element is parallel to Y-axis (only Y changes)
        elif dx < tolerance and dy > tolerance and dz < tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            element_counter += 1
        
        # Check if element is parallel to Z-axis (only Z changes)
        elif dx < tolerance and dy < tolerance and dz > tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            element_counter += 1

print(f"Total elements created: {len(elements)}")



# Add diagonal bracing elements on rectangular surfaces
# Connect nodes that share 2 out of 3 coordinates (are coplanar)

diagonal_tolerance = 1e-6
max_diagonal_length = min(np.sqrt(2 * L**2), max_element_length)  # Respect the max length constraint

# Create a set to track existing element connections (undirected)
existing_connections = set()
for idx in range(len(elements)):
    node1 = int(elements.iloc[idx]['Node1'])
    node2 = int(elements.iloc[idx]['Node2'])
    # Store as sorted tuple to make connection direction-agnostic
    existing_connections.add(tuple(sorted([node1, node2])))

for i in range(len(points)):
    node1 = points.iloc[i]
    node1_num = node1['Node Number']
    
    for j in range(i + 1, len(points)):
        node2 = points.iloc[j]
        node2_num = node2['Node Number']
        
        # Calculate differences
        dx = abs(node2['X'] - node1['X'])
        dy = abs(node2['Y'] - node1['Y'])
        dz = abs(node2['Z'] - node1['Z'])
        
        # Calculate actual distance between nodes
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Check if distance exceeds maximum allowed diagonal length
        if distance > max_diagonal_length + diagonal_tolerance:
            continue
        
        # Check if this connection already exists
        connection = tuple(sorted([int(node1_num), int(node2_num)]))
        if connection in existing_connections:
            continue
        
        # Check if nodes are coplanar (share 2 coordinates)
        # Case 1: X is constant (YZ plane) - both Y and Z change
        if dx < diagonal_tolerance and dy > diagonal_tolerance and dz > diagonal_tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            existing_connections.add(connection)
            element_counter += 1
        
        # Case 2: Y is constant (XZ plane) - both X and Z change
        elif dx > diagonal_tolerance and dy < diagonal_tolerance and dz > diagonal_tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            existing_connections.add(connection)
            element_counter += 1
        
        # Case 3: Z is constant (XY plane) - both X and Y change
        elif dx > diagonal_tolerance and dy > diagonal_tolerance and dz < diagonal_tolerance:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [int(node1_num)],
                'Node2': [int(node2_num)],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            existing_connections.add(connection)
            element_counter += 1

print(f"Total elements after adding diagonal bracing: {len(elements)}")



# Add 4 elements connecting Node 29 to nodes 8, 9, 15, 16
connecting_nodes = [8, 9, 15, 16]
for target_node in connecting_nodes:
    new_element = pd.DataFrame({
        'Element Number': [element_counter],
        'Node1': [29],
        'Node2': [target_node],
        'E': [210e9],
        'V': [0.3]
    })
    elements = pd.concat([elements, new_element], ignore_index=True)
    element_counter += 1

print(f"Total elements after adding connections from Node 29: {len(elements)}")


# Check for duplicate elements (same node pair regardless of order)
element_connections = []
for idx in range(len(elements)):
    node1 = int(elements.iloc[idx]['Node1'])
    node2 = int(elements.iloc[idx]['Node2'])
    # Store as sorted tuple to ensure (1,2) and (2,1) are treated as same
    connection = tuple(sorted([node1, node2]))
    element_connections.append(connection)

# Find duplicates
unique_connections = set(element_connections)
if len(element_connections) != len(unique_connections):
    print(f"\n⚠️  WARNING: Found duplicate elements!")
    print(f"Total elements: {len(element_connections)}")
    print(f"Unique connections: {len(unique_connections)}")
    print(f"Duplicates: {len(element_connections) - len(unique_connections)}")
    
    # Show which elements are duplicated
    from collections import Counter
    connection_counts = Counter(element_connections)
    duplicates = {conn: count for conn, count in connection_counts.items() if count > 1}
    print(f"\nDuplicate connections (Node1-Node2): count")
    for conn, count in duplicates.items():
        print(f"  Nodes {conn[0]}-{conn[1]}: appears {count} times")
else:
    print(f"\n✓ All {len(elements)} elements are unique (no duplicates)")


    # Add 6 additional specific elements connecting the specified node pairs
    additional_connections = [
        (9, 28),
        (16, 22),
        (23, 1),
        (2, 17),
        (1, 10),
        (2, 3)
    ]

    for node1, node2 in additional_connections:
        # Check if this connection already exists
        connection = tuple(sorted([node1, node2]))
        if connection not in existing_connections:
            new_element = pd.DataFrame({
                'Element Number': [element_counter],
                'Node1': [node1],
                'Node2': [node2],
                'E': [210e9],
                'V': [0.3]
            })
            elements = pd.concat([elements, new_element], ignore_index=True)
            existing_connections.add(connection)
            element_counter += 1
        else:
            print(f"Connection {node1}-{node2} already exists, skipping.")

    print(f"Total elements after adding 6 specific connections: {len(elements)}")








# Loop through each element in the elements DataFrame
element_lengths = []

for i in range(len(elements)):
    # Get node coordinates for this element
    node1_idx = elements.iloc[i]['Node1'] - 1
    node2_idx = elements.iloc[i]['Node2'] - 1

    x1, y1, z1 = points.loc[node1_idx, ['X', 'Y', 'Z']]
    x2, y2, z2 = points.loc[node2_idx, ['X', 'Y', 'Z']]

    # Calculate element length
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    element_lengths.append(length)

    print(f"Element {elements.iloc[i]['Element Number']}: Length = {length:.4f}")

# Add 'Element Length' column to elements DataFrame
elements['Element Length'] = element_lengths

# Add 'Element Cross Section' column (empty for now)
elements['Element Cross Section'] = np.nan

print(f"\nAdded 'Element Length' and 'Element Cross Section' columns to elements DataFrame")

# Set cross-sectional areas based on element length
for i in range(len(elements)):
    length = elements.iloc[i]['Element Length']
    
    if length < 1.9:  # Straight elements
        elements.loc[i, 'Element Cross Section'] = 1.5 * A_0
    elif length > 2:  # Diagonal elements
        elements.loc[i, 'Element Cross Section'] = 0.5 * A_0
    # Elements between 1.9 and 2 will remain NaN

print(f"\nCross-sectional areas assigned:")
print(f"  Straight elements (L < 1.9): {1.5 * A_0:.6f} m²")
print(f"  Diagonal elements (L > 2): {0.5 * A_0:.6f} m²")

# Count elements by type
num_straight = len(elements[elements['Element Length'] < 1.9])
num_diagonal = len(elements[elements['Element Length'] > 2])
num_unassigned = len(elements[elements['Element Cross Section'].isna()])

print(f"\nElement count by type:")
print(f"  Straight: {num_straight}")
print(f"  Diagonal: {num_diagonal}")
print(f"  Unassigned: {num_unassigned}")


# Display the elements DataFrame in an interactive scrollable window (after adding all columns)
display_matrix_table(elements, "Elements DataFrame")





























################### Rotate Structure Around Y Axis #####################




# Optionally rotate the structure about Y before plotting/analysis
# Set angle_deg to your desired rotation (degrees). Leave 0 for no rotation.
angle_deg = - phi # e.g., 30.0 to rotate 30 degrees about +Y
# You can also change the pivot point if needed: origin=(x0, y0, z0)
points = rotate_points_around_y(points, angle_deg, origin=(A, 0.0, 0.0))












































####################Plot Points in 3D with Plotly (Interactive)##########################

# Create interactive Plotly 3D scatter plot for nodes
fig_plotly_nodes = go.Figure()

# Add nodes as scatter points with hover info
fig_plotly_nodes.add_trace(go.Scatter3d(
    x=points['X'],
    y=points['Y'],
    z=points['Z'],
    mode='markers+text',
    marker=dict(size=8, color='blue', opacity=0.8),
    text=[f"Node {int(n)}" for n in points['Node Number']],
    textposition='top center',
    textfont=dict(size=10),
    hovertemplate='<b>Node %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
    name='Nodes'
))

# Set equal aspect ratio and layout
fig_plotly_nodes.update_layout(
    title='Interactive 3D Nodes Plot',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',  # Equal aspect ratio
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1000,
    height=800,
    hovermode='closest'
)

fig_plotly_nodes.show()











####################DEFINING FORCES AND SUPPORTS##########################

# Initialize forces DataFrame
# 3 force components (Fx, Fy, Fz) per node
num_force_rows = 3 * len(points)

# Create force labels (Fx1, Fy1, Fz1, Fx2, Fy2, Fz2, ...)
force_labels = [f"F{n}{axis}" for n in range(1, len(points) + 1) for axis in ("x", "y", "z")]

# Initialize DataFrame with NaN values
forces = pd.DataFrame({
    'Variable': force_labels,
    'Value (Newton)': [np.nan] * num_force_rows
})

print(f"Forces DataFrame created with {len(forces)} rows (3 DOF × {len(points)} nodes)")


# Display the forces DataFrame in an interactive scrollable window
display_matrix_table(forces, "Forces DataFrame")



# Assign -2000 N force in Z direction at node 29
# Node 29 corresponds to force index for Fz29
# Calculate the row index: (node_number - 1) * 3 + 2 (where 2 is for z-component)
node_29_fz_index = (29 - 1) * 3 + 2

# Set the force value
forces.loc[node_29_fz_index, 'Value (Newton)'] = -2000

print(f"\nForce assigned: Fz29 = -2000 N (row index {node_29_fz_index})")




# Initialize displacements DataFrame
# 3 displacement components (Ux, Uy, Uz) per node
num_displacement_rows = 3 * len(points)

# Create displacement labels (U1x, U1y, U1z, U2x, U2y, U2z, ...)
displacement_labels = [f"U{n}{axis}" for n in range(1, len(points) + 1) for axis in ("x", "y", "z")]

# Initialize DataFrame with NaN values
displacements = pd.DataFrame({
    'Variable': displacement_labels,
    'Value (m)': [np.nan] * num_displacement_rows
})

print(f"Displacements DataFrame created with {len(displacements)} rows (3 DOF × {len(points)} nodes)")

# Display the displacements DataFrame in an interactive scrollable window
display_matrix_table(displacements, "Displacements DataFrame")


# Set all displacements (Ux, Uy, Uz) to zero for nodes 1 and 2
for node_num in [1, 2]:
    # Calculate the row indices for this node's displacements
    ux_idx = (node_num - 1) * 3 + 0  # x-component
    uy_idx = (node_num - 1) * 3 + 1  # y-component
    uz_idx = (node_num - 1) * 3 + 2  # z-component
    
    # Set displacement values to zero
    displacements.loc[ux_idx, 'Value (m)'] = 0.0
    displacements.loc[uy_idx, 'Value (m)'] = 0.0
    displacements.loc[uz_idx, 'Value (m)'] = 0.0

print(f"\nDisplacements set to zero for nodes 1 and 2 (fixed supports)")





















################Plot Elements in 3D with Plotly (Interactive)##########################

# Specify elements to highlight (change this list to highlight different elements)
highlighted_elements = [7]  # e.g., [7, 15, 23] to highlight multiple elements
highlight_color = 'red'
highlight_width = 6

# Create interactive Plotly 3D plot for structure with elements
fig_plotly_elements = go.Figure()

# Add regular elements (lines between nodes)
edge_x = []
edge_y = []
edge_z = []

for idx in range(len(elements)):
    elem_num = int(elements.iloc[idx]['Element Number'])
    
    # Skip highlighted elements in this loop
    if elem_num in highlighted_elements:
        continue
    
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    # Get coordinates
    x1, y1, z1 = points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    edge_x.extend([x1, x2, None])  # None creates a break between lines
    edge_y.extend([y1, y2, None])
    edge_z.extend([z1, z2, None])

# Add regular elements as lines
fig_plotly_elements.add_trace(go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    line=dict(color='black', width=3),
    hoverinfo='skip',
    name='Elements'
))

# Add highlighted elements separately with different styling
for elem_num in highlighted_elements:
    # Find the element in the dataframe
    elem_row = elements[elements['Element Number'] == elem_num]
    if len(elem_row) == 0:
        continue
    
    elem_row = elem_row.iloc[0]
    node1_idx = int(elem_row['Node1']) - 1
    node2_idx = int(elem_row['Node2']) - 1
    
    # Get coordinates
    x1, y1, z1 = points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    # Add as a separate trace
    fig_plotly_elements.add_trace(go.Scatter3d(
        x=[x1, x2],
        y=[y1, y2],
        z=[z1, z2],
        mode='lines',
        line=dict(color=highlight_color, width=highlight_width),
        hovertemplate=f'<b>Element {elem_num} (Highlighted)</b><br>Node {int(elem_row["Node1"])} → Node {int(elem_row["Node2"])}<extra></extra>',
        name=f'Element {elem_num}',
        showlegend=True
    ))

# Add element numbers at midpoints
element_midpoints_x = []
element_midpoints_y = []
element_midpoints_z = []
element_labels = []

for idx in range(len(elements)):
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    # Get coordinates
    x1, y1, z1 = points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    # Calculate midpoint
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    mid_z = (z1 + z2) / 2
    
    element_midpoints_x.append(mid_x)
    element_midpoints_y.append(mid_y)
    element_midpoints_z.append(mid_z)
    element_labels.append(f"E{int(elements.iloc[idx]['Element Number'])}")

# Add element numbers as text labels
fig_plotly_elements.add_trace(go.Scatter3d(
    x=element_midpoints_x,
    y=element_midpoints_y,
    z=element_midpoints_z,
    mode='text',
    text=element_labels,
    textfont=dict(size=8, color='green'),
    hovertemplate='<b>%{text}</b><br>Midpoint: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
    name='Element Labels',
    showlegend=True
))

# Add nodes as scatter points
fig_plotly_elements.add_trace(go.Scatter3d(
    x=points['X'],
    y=points['Y'],
    z=points['Z'],
    mode='markers+text',
    marker=dict(size=8, color='red', opacity=0.9),
    text=[f"Node {int(n)}" for n in points['Node Number']],
    textposition='top center',
    textfont=dict(size=9),
    hovertemplate='<b>Node %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
    name='Nodes'
))

# ============ ADD FORCE VISUALIZATION ============
# Extract forces from the forces DataFrame and visualize them
applied_forces = []

# Parse the forces DataFrame to extract non-zero forces
for node_num in range(1, len(points) + 1):
    # Get force indices for this node
    fx_idx = (node_num - 1) * 3 + 0  # x-component
    fy_idx = (node_num - 1) * 3 + 1  # y-component
    fz_idx = (node_num - 1) * 3 + 2  # z-component
    
    # Get force values (convert NaN to 0)
    fx = forces.loc[fx_idx, 'Value (Newton)']
    fy = forces.loc[fy_idx, 'Value (Newton)']
    fz = forces.loc[fz_idx, 'Value (Newton)']
    
    fx = 0 if pd.isna(fx) else fx
    fy = 0 if pd.isna(fy) else fy
    fz = 0 if pd.isna(fz) else fz
    
    # Only add if at least one component is non-zero
    if abs(fx) > 1e-6 or abs(fy) > 1e-6 or abs(fz) > 1e-6:
        magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
        applied_forces.append((node_num, [fx, fy, fz], magnitude))

print(f"\nVisualizing {len(applied_forces)} applied forces")

# Visualization settings
force_scale = 0.0008  # Scale factor for arrow length (adjust for visibility)
arrow_color = 'magenta'
arrow_width = 8
cone_size_ratio = 0.25  # Size of arrowhead relative to total length

for idx, (node_num, force_vec, magnitude) in enumerate(applied_forces):
    if node_num > len(points):
        continue
    
    # Get node position
    node_pos = points[points['Node Number'] == node_num].iloc[0]
    x0, y0, z0 = node_pos['X'], node_pos['Y'], node_pos['Z']
    
    # Calculate arrow direction (force direction)
    fx, fy, fz = force_vec
    
    # Scale the arrow based on magnitude and scale factor
    arrow_len = magnitude * force_scale
    dx_total = fx / magnitude * arrow_len if magnitude > 0 else 0
    dy_total = fy / magnitude * arrow_len if magnitude > 0 else 0
    dz_total = fz / magnitude * arrow_len if magnitude > 0 else 0
    
    # Calculate shaft and cone portions
    shaft_ratio = 1 - cone_size_ratio
    dx_shaft = dx_total * shaft_ratio
    dy_shaft = dy_total * shaft_ratio
    dz_shaft = dz_total * shaft_ratio
    
    dx_cone = dx_total * cone_size_ratio
    dy_cone = dy_total * cone_size_ratio
    dz_cone = dz_total * cone_size_ratio
    
    # Add force line (shaft of arrow) - make it thicker and add to legend
    show_in_legend = (idx == 0)  # Only show first force in legend
    legend_label = f"External Force: {magnitude:.0f} N" if show_in_legend else None
    
    fig_plotly_elements.add_trace(go.Scatter3d(
        x=[x0, x0 + dx_shaft],
        y=[y0, y0 + dy_shaft],
        z=[z0, z0 + dz_shaft],
        mode='lines',
        line=dict(color=arrow_color, width=arrow_width),
        showlegend=show_in_legend,
        name=legend_label if show_in_legend else None,
        hovertemplate=f'<b>External Force on Node {node_num}</b><br>Magnitude: {magnitude:.0f} N<br>Fx: {fx:.0f} N<br>Fy: {fy:.0f} N<br>Fz: {fz:.0f} N<extra></extra>'
    ))
    
    # Add force arrow using cone (3D arrowhead) at the end of shaft
    fig_plotly_elements.add_trace(go.Cone(
        x=[x0 + dx_shaft],
        y=[y0 + dy_shaft],
        z=[z0 + dz_shaft],
        u=[dx_cone],
        v=[dy_cone],
        w=[dz_cone],
        colorscale=[[0, arrow_color], [1, arrow_color]],
        showscale=False,
        sizemode='absolute',
        sizeref=arrow_len * 0.4,
        anchor='tail',
        showlegend=False,
        hoverinfo='skip'
    ))

# Set equal aspect ratio and layout
fig_plotly_elements.update_layout(
    title=f'Interactive 3D Structure Plot ({len(elements)} Elements, {len(points)} Nodes)',
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',  # Equal aspect ratio
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=900,
    hovermode='closest',
    showlegend=True
)

fig_plotly_elements.show()








####################Old Matplotlib Plots (Commented Out)##########################

# Uncomment these if you want the old matplotlib plots as well

# # Convert to numpy array for compatibility with existing plotting code
# points_array = points[['X', 'Y', 'Z']].to_numpy()

# # Create interactive 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot points
# ax.scatter(points['X'], points['Y'], points['Z'], color='blue', marker='o', s=100)

# # Add node numbers as labels
# for i, row in points.iterrows():
#     ax.text(row['X'], row['Y'], row['Z'], 
#             f'  Node {row["Node Number"]}', 
#             fontsize=10)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Points Plot')
# ax.grid(True)

# # Enforce equal axis scaling for accurate geometry perception
# set_3d_axes_equal(ax, points['X'].to_numpy(), points['Y'].to_numpy(), points['Z'].to_numpy())

# plt.show()


# # Create interactive 3D plot for elements
# fig2 = plt.figure(figsize=(10, 8))
# ax2 = fig2.add_subplot(111, projection='3d')

# # Plot points
# ax2.scatter(points['X'], points['Y'], points['Z'], color='blue', marker='o', s=100)

# # Add node numbers as labels
# for i, row in points.iterrows():
#     ax2.text(row['X'], row['Y'], row['Z'], 
#             f'  Node {row["Node Number"]}', 
#             fontsize=10)

# # Plot the connections between points
# for i in range(len(elements)):
#     n1 = elements['Node1'][i] - 1  # Adjust for 0-based indexing
#     n2 = elements['Node2'][i] - 1
#     ax2.plot([points['X'].iloc[n1], points['X'].iloc[n2]], 
#              [points['Y'].iloc[n1], points['Y'].iloc[n2]],
#              [points['Z'].iloc[n1], points['Z'].iloc[n2]], 'k-', linewidth=2)

# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('3D Elements Plot')
# ax2.grid(True)

# # Enforce equal axis scaling for accurate geometry perception
# set_3d_axes_equal(ax2, points['X'].to_numpy(), points['Y'].to_numpy(), points['Z'].to_numpy())

# plt.show()































































# Create empty stiffness matrix (3 DOF per node: x, y, z)
num_nodes = len(points)
n_dof = 3 * num_nodes
row_labels = [f"F{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
col_labels = [f"V{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
stiffness_matrix = pd.DataFrame(np.zeros((n_dof, n_dof)), index=row_labels, columns=col_labels)


# Display the stiffness matrix in an interactive scrollable window using tkinter
# Display the stiffness matrix
display_matrix_table(stiffness_matrix, "Stiffness Matrix")






# Loop through all elements and assemble global stiffness matrix
for element_idx in range(len(elements)):
    # Get element properties
    elem = elements.iloc[element_idx]
    node1_num = int(elem['Node1'])
    node2_num = int(elem['Node2'])
    E = elem['E']  # Young's modulus
    A = elem['Element Cross Section']  # Cross-sectional area
    l_e = elem['Element Length']  # Element length

    # Get node coordinates
    node1 = points[points['Node Number'] == node1_num].iloc[0]
    node2 = points[points['Node Number'] == node2_num].iloc[0]

    X_i = node1['X']
    Y_i = node1['Y']
    Z_i = node1['Z']

    X_j = node2['X']
    Y_j = node2['Y']
    Z_j = node2['Z']

    # Calculate direction cosines
    l_y = (X_j - X_i) / l_e  # cos(x, X)
    m_y = (Y_j - Y_i) / l_e  # cos(x, Y)
    n_y = (Z_j - Z_i) / l_e  # cos(x, Z)

    # Define the local stiffness matrix K_e for the element
    K_e = (A * E / l_e) * np.array([
        [l_y**2,        l_y*m_y,      l_y*n_y,     -l_y**2,       -l_y*m_y,     -l_y*n_y    ],
        [l_y*m_y,       m_y**2,       m_y*n_y,     -l_y*m_y,      -m_y**2,      -m_y*n_y    ],
        [l_y*n_y,       m_y*n_y,      n_y**2,      -l_y*n_y,      -m_y*n_y,     -n_y**2     ],
        [-l_y**2,       -l_y*m_y,     -l_y*n_y,     l_y**2,        l_y*m_y,      l_y*n_y    ],
        [-l_y*m_y,      -m_y**2,      -m_y*n_y,     l_y*m_y,       m_y**2,       m_y*n_y    ],
        [-l_y*n_y,      -m_y*n_y,     -n_y**2,      l_y*n_y,       m_y*n_y,      n_y**2     ]
    ])

    # Map local DOFs to global DOFs
    global_dofs = [
        (node1_num - 1) * 3 + 0,  # Node 1 x
        (node1_num - 1) * 3 + 1,  # Node 1 y
        (node1_num - 1) * 3 + 2,  # Node 1 z
        (node2_num - 1) * 3 + 0,  # Node 2 x
        (node2_num - 1) * 3 + 1,  # Node 2 y
        (node2_num - 1) * 3 + 2   # Node 2 z
    ]

    # Add K_e values to global stiffness matrix
    for i, global_i in enumerate(global_dofs):
        for j, global_j in enumerate(global_dofs):
            stiffness_matrix.iloc[global_i, global_j] += K_e[i, j]

print(f"\nGlobal stiffness matrix assembly complete.")
print(f"Processed {len(elements)} elements.")

# Display the assembled global stiffness matrix in an interactive scrollable window
display_matrix_table(stiffness_matrix, "Assembled Global Stiffness Matrix")

# Optionally, also print summary statistics
print(f"\nGlobal Stiffness Matrix Summary:")
print(f"  Size: {stiffness_matrix.shape[0]} × {stiffness_matrix.shape[1]}")
print(f"  Max value: {stiffness_matrix.values.max():.2e}")
print(f"  Min value: {stiffness_matrix.values.min():.2e}")
print(f"  Non-zero elements: {np.count_nonzero(stiffness_matrix.values)}")





