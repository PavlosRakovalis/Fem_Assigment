"""
3D Finite Element Method (FEM) Analysis with BEAM ELEMENTS

This script implements a complete FEM analysis using 3D beam elements that can handle:
  - Axial forces (tension/compression along element axis)
  - Bending moments (about two perpendicular axes)
  - Torsion (twisting about element axis)
  - Shear forces

Element Formulation:
  - 6 DOF per node: 3 translations (ux, uy, uz) + 3 rotations (rx, ry, rz)
  - 12×12 local stiffness matrix (2 nodes × 6 DOF)
  - Euler-Bernoulli beam theory (neglects shear deformation)
  - Transformation from local to global coordinates using direction cosines

Section Properties (assuming square cross-section with height h):
  - Area: A = h²
  - Moment of inertia: I_y = I_z = h⁴/12
  - Torsional constant: J ≈ 0.141×h⁴
  - Shear modulus: G = E/(2(1+ν))

Note: This is an upgraded version that replaces the original truss elements
      (3 DOF per node, axial forces only) with full beam elements.
"""

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
    Falls back to console/HTML output in headless environments.
    """
    import os
    
    # Try to open a Tkinter window; if DISPLAY is not set or Tk can't start,
    # fall back to console/HTML output so the code works in headless environments.
    try:
        # Quick check: on many headless systems DISPLAY is unset
        if os.environ.get("DISPLAY", "") == "":
            raise tk.TclError("no DISPLAY")

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

    except (tk.TclError, RuntimeError) as e:
        # Headless fallback: print to console and save an HTML copy
        print(f"\n{'='*80}")
        print(f"[{title}] GUI not available (headless environment).")
        print(f"Displaying matrix in console and saving to HTML file...")
        print(f"{'='*80}\n")
        
        # Print summary
        print(f"Matrix size: {df.shape[0]} × {df.shape[1]}")
        print(f"\nFirst few rows/columns:\n")
        
        # Show a preview (first 10 rows and columns)
        preview = df.iloc[:10, :10] if df.shape[0] > 10 or df.shape[1] > 10 else df
        print(preview.to_string())
        
        if df.shape[0] > 10 or df.shape[1] > 10:
            print(f"\n... (showing first 10×10 subset of {df.shape[0]}×{df.shape[1]} matrix)")
        
        # Save full HTML for viewing/download in organized folder
        output_dir = os.path.join(os.getcwd(), "output_html")
        os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
        
        html_filename = f"{title.replace(' ', '_')}.html"
        html_path = os.path.join(output_dir, html_filename)
        try:
            df.to_html(html_path, index=True)
            print(f"\n✓ Full matrix saved to: {html_path}")
            print(f"  (You can download this file to view the complete matrix)\n")
        except Exception as ex:
            print(f"\n⚠ Could not save HTML file: {ex}\n")






























####################################   DEFINING NODES   ######################




# Creating a DataFrame with node numbers and coordinates
#it is empty for now
points = pd.DataFrame({
    'Node Number': [],
    'X': [],
    'Y': [],
    'Z': []
})



L = 1.5 * 1.13  # Length in meters
A = 1.2 * 1.69  # Distance in meters
phi = 60 + 9.1  # degrees
# A_0 represents the base cross-sectional area parameter
# Adjusted to achieve target displacement of ~60mm under 2000N load
A_0 = (0.5 + 0.13) / 6.0  # = 0.105 (base value for area calculation in cm²)


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



# DIAGONAL BRACING REMOVED
# Previously, diagonal elements were added on rectangular surfaces
# This section has been commented out to remove all diagonal elements

# # Add diagonal bracing elements on rectangular surfaces
# # Connect nodes that share 2 out of 3 coordinates (are coplanar)
# 
# diagonal_tolerance = 1e-6
# max_diagonal_length = min(np.sqrt(2 * L**2), max_element_length)  # Respect the max length constraint
# 
# # Create a set to track existing element connections (undirected)
# existing_connections = set()
# for idx in range(len(elements)):
#     node1 = int(elements.iloc[idx]['Node1'])
#     node2 = int(elements.iloc[idx]['Node2'])
#     # Store as sorted tuple to make connection direction-agnostic
#     existing_connections.add(tuple(sorted([node1, node2])))
# 
# for i in range(len(points)):
#     node1 = points.iloc[i]
#     node1_num = node1['Node Number']
#     
#     for j in range(i + 1, len(points)):
#         node2 = points.iloc[j]
#         node2_num = node2['Node Number']
#         
#         # Calculate differences
#         dx = abs(node2['X'] - node1['X'])
#         dy = abs(node2['Y'] - node1['Y'])
#         dz = abs(node2['Z'] - node1['Z'])
#         
#         # Calculate actual distance between nodes
#         distance = np.sqrt(dx**2 + dy**2 + dz**2)
#         
#         # Check if distance exceeds maximum allowed diagonal length
#         if distance > max_diagonal_length + diagonal_tolerance:
#             continue
#         
#         # Check if this connection already exists
#         connection = tuple(sorted([int(node1_num), int(node2_num)]))
#         if connection in existing_connections:
#             continue
#         
#         # Check if nodes are coplanar (share 2 coordinates)
#         # Case 1: X is constant (YZ plane) - both Y and Z change
#         if dx < diagonal_tolerance and dy > diagonal_tolerance and dz > diagonal_tolerance:
#             new_element = pd.DataFrame({
#                 'Element Number': [element_counter],
#                 'Node1': [int(node1_num)],
#                 'Node2': [int(node2_num)],
#                 'E': [210e9],
#                 'V': [0.3]
#             })
#             elements = pd.concat([elements, new_element], ignore_index=True)
#             existing_connections.add(connection)
#             element_counter += 1
#         
#         # Case 2: Y is constant (XZ plane) - both X and Z change
#         elif dx > diagonal_tolerance and dy < diagonal_tolerance and dz > diagonal_tolerance:
#             new_element = pd.DataFrame({
#                 'Element Number': [element_counter],
#                 'Node1': [int(node1_num)],
#                 'Node2': [int(node2_num)],
#                 'E': [210e9],
#                 'V': [0.3]
#             })
#             elements = pd.concat([elements, new_element], ignore_index=True)
#             existing_connections.add(connection)
#             element_counter += 1
#         
#         # Case 3: Z is constant (XY plane) - both X and Y change
#         elif dx > diagonal_tolerance and dy > diagonal_tolerance and dz < diagonal_tolerance:
#             new_element = pd.DataFrame({
#                 'Element Number': [element_counter],
#                 'Node1': [int(node1_num)],
#                 'Node2': [int(node2_num)],
#                 'E': [210e9],
#                 'V': [0.3]
#             })
#             elements = pd.concat([elements, new_element], ignore_index=True)
#             existing_connections.add(connection)
#             element_counter += 1

print(f"Total elements (diagonal bracing removed): {len(elements)}")

# Create a set to track existing element connections (undirected)
existing_connections = set()
for idx in range(len(elements)):
    node1 = int(elements.iloc[idx]['Node1'])
    node2 = int(elements.iloc[idx]['Node2'])
    # Store as sorted tuple to make connection direction-agnostic
    existing_connections.add(tuple(sorted([node1, node2])))


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
    existing_connections.add(tuple(sorted([29, target_node])))
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
# A_0 is dimensionless, cross-sections are in cm² and need conversion to m²
# Using A_0 divided by factors (not multiplied) based on element type
for i in range(len(elements)):
    length = elements.iloc[i]['Element Length']
    
    if length < 1.9:  # Straight elements
        elements.loc[i, 'Element Cross Section'] = (A_0 / 1.5) * 1e-4  # (A_0/1.5) cm² to m²
    elif length > 2:  # Diagonal elements  
        elements.loc[i, 'Element Cross Section'] = (A_0 / 0.5) * 1e-4  # (A_0/0.5) cm² to m²
    # Elements between 1.9 and 2 will remain NaN

print(f"\nCross-sectional areas assigned:")
print(f"  Straight elements (L < 1.9): {A_0/1.5:.4f} cm² = {(A_0/1.5) * 1e-4:.6e} m²")
print(f"  Diagonal elements (L > 2): {A_0/0.5:.4f} cm² = {(A_0/0.5) * 1e-4:.6e} m²")

# Count elements by type
num_straight = len(elements[elements['Element Length'] < 1.9])
num_diagonal = len(elements[elements['Element Length'] > 2])
num_unassigned = len(elements[elements['Element Cross Section'].isna()])

print(f"\nElement count by type:")
print(f"  Straight: {num_straight}")
print(f"  Diagonal: {num_diagonal}")
print(f"  Unassigned: {num_unassigned}")

# ⚠️ CRITICAL CHECK: Elements with NaN cross section will cause HUGE displacements!
if num_unassigned > 0:
    print(f"\n⚠️⚠️⚠️  CRITICAL ERROR: {num_unassigned} elements have NO cross section! ⚠️⚠️⚠️")
    print(f"These elements will have infinite flexibility (A=NaN → K=NaN)")
    print(f"\nElements with missing cross section:")
    nan_elements = elements[elements['Element Cross Section'].isna()]
    for idx in nan_elements.index[:10]:  # Show first 10
        elem = elements.iloc[idx]
        print(f"  Element {int(elem['Element Number'])}: Length = {elem['Element Length']:.4f} m, A = NaN")
    if len(nan_elements) > 10:
        print(f"  ... and {len(nan_elements) - 10} more")
    print(f"\n⚠️  FIX: Assign cross sections to ALL elements before continuing! ⚠️")


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

# ========== PERFORMANCE OPTIMIZATION ==========
# To restore all plots and features, ask Copilot:
# "Undo the Plotly performance optimizations - restore element labels, all 3 plots, and full hover info"
# ==============================================

# fig_plotly_nodes.show()  # Disabled for performance - too slow in browser











####################DEFINING FORCES AND SUPPORTS##########################

# Initialize forces DataFrame for BEAM ELEMENTS
# 6 load components per node: 3 forces (Fx, Fy, Fz) + 3 moments (Mx, My, Mz)
num_force_rows = 6 * len(points)

# Create force labels (Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2, ...)
force_labels = [f"{load_type}{n}" 
                for n in range(1, len(points) + 1) 
                for load_type in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]

# Initialize DataFrame with NaN values
forces = pd.DataFrame({
    'Variable': force_labels,
    'Value (N or N·m)': [np.nan] * num_force_rows
})

print(f"Forces DataFrame created with {len(forces)} rows (6 DOF × {len(points)} nodes)")
print(f"  - Each node has 3 force components (Fx, Fy, Fz) and 3 moment components (Mx, My, Mz)")


# Display the forces DataFrame in an interactive scrollable window
display_matrix_table(forces, "Forces DataFrame")



# Assign -2000 N force in Z direction at node 29
# Node 29 corresponds to force index for Fz29
# Calculate the row index: (node_number - 1) * 6 + 2 (where 2 is for z-component)
node_29_fz_index = (29 - 1) * 6 + 2

# Set the force value
forces.loc[node_29_fz_index, 'Value (N or N·m)'] = -2000

print(f"\nForce assigned: Fz29 = -2000 N (row index {node_29_fz_index})")
print(f"  Note: For beam elements, you can also apply moments (Mx, My, Mz) at any node")




# Initialize displacements DataFrame for BEAM ELEMENTS
# 6 DOF per node: 3 translations (Ux, Uy, Uz) + 3 rotations (Rx, Ry, Rz)
num_displacement_rows = 6 * len(points)

# Create displacement labels (Ux1, Uy1, Uz1, Rx1, Ry1, Rz1, Ux2, Uy2, Uz2, Rx2, Ry2, Rz2, ...)
displacement_labels = [f"{dof_type}{n}" 
                       for n in range(1, len(points) + 1) 
                       for dof_type in ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")]

# Initialize DataFrame with NaN values
displacements = pd.DataFrame({
    'Variable': displacement_labels,
    'Value (m or rad)': [np.nan] * num_displacement_rows
})

print(f"Displacements DataFrame created with {len(displacements)} rows (6 DOF × {len(points)} nodes)")
print(f"  - Each node has 3 translations (Ux, Uy, Uz) and 3 rotations (Rx, Ry, Rz)")

# Display the displacements DataFrame in an interactive scrollable window
display_matrix_table(displacements, "Displacements DataFrame")


# Set all DOFs (Ux, Uy, Uz, Rx, Ry, Rz) to zero for nodes 1 and 2
# For beam elements, this creates fully fixed supports (no translation or rotation)
for node_num in [1, 2]:
    # Calculate the row indices for this node's 6 DOF
    ux_idx = (node_num - 1) * 6 + 0  # x-translation
    uy_idx = (node_num - 1) * 6 + 1  # y-translation
    uz_idx = (node_num - 1) * 6 + 2  # z-translation
    rx_idx = (node_num - 1) * 6 + 3  # x-rotation
    ry_idx = (node_num - 1) * 6 + 4  # y-rotation
    rz_idx = (node_num - 1) * 6 + 5  # z-rotation
    
    # Set all displacement and rotation values to zero (fully fixed)
    displacements.loc[ux_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[uy_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[uz_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[rx_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[ry_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[rz_idx, 'Value (m or rad)'] = 0.0

print(f"\nAll 6 DOF set to zero for nodes 1 and 2 (fully fixed supports)")


# Set all DOFs (Ux, Uy, Uz, Rx, Ry, Rz) to zero for nodes 19, 25, 22, 28
# For beam elements, this creates fully fixed supports (no translation or rotation)
for node_num in [19, 25, 22, 28]:
    # Calculate the row indices for this node's 6 DOF
    ux_idx = (node_num - 1) * 6 + 0  # x-translation
    uy_idx = (node_num - 1) * 6 + 1  # y-translation
    uz_idx = (node_num - 1) * 6 + 2  # z-translation
    rx_idx = (node_num - 1) * 6 + 3  # x-rotation
    ry_idx = (node_num - 1) * 6 + 4  # y-rotation
    rz_idx = (node_num - 1) * 6 + 5  # z-rotation
    
    # Set all displacement and rotation values to zero (fully fixed)
    displacements.loc[ux_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[uy_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[uz_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[rx_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[ry_idx, 'Value (m or rad)'] = 0.0
    displacements.loc[rz_idx, 'Value (m or rad)'] = 0.0

print(f"\nAll 6 DOF set to zero for nodes 19, 25, 22, 28 (additional fully fixed supports)")





















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
# DISABLED FOR PERFORMANCE - Element labels are heavy to render
# fig_plotly_elements.add_trace(go.Scatter3d(
#     x=element_midpoints_x,
#     y=element_midpoints_y,
#     z=element_midpoints_z,
#     mode='text',
#     text=element_labels,
#     textfont=dict(size=8, color='green'),
#     hovertemplate='<b>%{text}</b><br>Midpoint: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
#     name='Element Labels',
#     showlegend=True
# ))

# Add nodes as scatter points (reduced size for performance)
fig_plotly_elements.add_trace(go.Scatter3d(
    x=points['X'],
    y=points['Y'],
    z=points['Z'],
    mode='markers+text',
    marker=dict(size=5, color='red', opacity=0.9),  # Reduced from 8 to 5
    text=[f"Node {int(n)}" for n in points['Node Number']],
    textposition='top center',
    textfont=dict(size=8),  # Reduced from 9 to 8
    hoverinfo='skip',  # Simplified hover for performance
    name='Nodes'
))

# ============ ADD FORCE VISUALIZATION ============
# Extract forces from the forces DataFrame and visualize them
applied_forces = []

# Parse the forces DataFrame to extract non-zero forces (beam elements have 6 DOF per node)
for node_num in range(1, len(points) + 1):
    # Get force indices for this node (6 DOF: 3 forces + 3 moments)
    fx_idx = (node_num - 1) * 6 + 0  # x-force
    fy_idx = (node_num - 1) * 6 + 1  # y-force
    fz_idx = (node_num - 1) * 6 + 2  # z-force
    # Note: Moments (Mx, My, Mz) at indices +3, +4, +5 are not visualized as arrows
    
    # Get force values (convert NaN to 0)
    fx = forces.loc[fx_idx, 'Value (N or N·m)']
    fy = forces.loc[fy_idx, 'Value (N or N·m)']
    fz = forces.loc[fz_idx, 'Value (N or N·m)']
    
    fx = 0 if pd.isna(fx) else fx
    fy = 0 if pd.isna(fy) else fy
    fz = 0 if pd.isna(fz) else fz
    
    # Only add if at least one force component is non-zero
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

# fig_plotly_elements.show()  # Disabled for performance - too slow in browser








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































































# Create empty stiffness matrix for BEAM ELEMENTS (6 DOF per node: ux, uy, uz, rx, ry, rz)
num_nodes = len(points)
n_dof = 6 * num_nodes
row_labels = [f"F{n}{dof}" for n in range(1, num_nodes + 1) for dof in ("x", "y", "z", "rx", "ry", "rz")]
col_labels = [f"V{n}{dof}" for n in range(1, num_nodes + 1) for dof in ("x", "y", "z", "rx", "ry", "rz")]
stiffness_matrix = pd.DataFrame(np.zeros((n_dof, n_dof)), index=row_labels, columns=col_labels)

print(f"\nInitialized global stiffness matrix: {n_dof} × {n_dof} (6 DOF × {num_nodes} nodes)")
print(f"  Each node has 6 DOF: 3 translations (x,y,z) + 3 rotations (rx,ry,rz)")


# Display the stiffness matrix in an interactive scrollable window using tkinter
# Display the stiffness matrix
display_matrix_table(stiffness_matrix, "Stiffness Matrix")






# Loop through all elements and assemble global stiffness matrix using BEAM ELEMENTS
for element_idx in range(len(elements)):
    # Get element properties
    elem = elements.iloc[element_idx]
    node1_num = int(elem['Node1'])
    node2_num = int(elem['Node2'])
    E = elem['E']  # Young's modulus (Pa)
    A = elem['Element Cross Section']  # Cross-sectional area (m²)
    l_e = elem['Element Length']  # Element length (m)
    
    # Calculate section height from area (assuming square cross-section: A = h²)
    h = np.sqrt(A)  # Cross-section height (m)

    # Get node coordinates
    node1 = points[points['Node Number'] == node1_num].iloc[0]
    node2 = points[points['Node Number'] == node2_num].iloc[0]

    X_i = node1['X']
    Y_i = node1['Y']
    Z_i = node1['Z']

    X_j = node2['X']
    Y_j = node2['Y']
    Z_j = node2['Z']

    # Calculate beam section properties (assuming square cross-section)
    # For square section: I = h^4 / 12
    I_y = h**4 / 12  # Moment of inertia about local y-axis (m⁴)
    I_z = h**4 / 12  # Moment of inertia about local z-axis (m⁴)
    
    # Torsional constant for square section: J ≈ 0.141 * h^4
    J = 0.141 * h**4  # Torsional constant (m⁴)
    
    # Shear modulus G = E / (2 * (1 + ν))
    # Assuming Poisson's ratio ν = 0.3 for steel
    nu = 0.3
    G = E / (2 * (1 + nu))

    # Debug first few elements to check stiffness values
    if element_idx < 3:
        print(f"\n🔍 BEAM Element {element_idx+1} stiffness check:")
        print(f"   E = {E:.2e} Pa")
        print(f"   A = {A:.6e} m²")
        print(f"   h = {h:.6f} m")
        print(f"   L = {l_e:.4f} m")
        print(f"   I_y = I_z = {I_y:.6e} m⁴")
        print(f"   J = {J:.6e} m⁴")
        print(f"   Axial stiffness (EA/L) = {(A*E/l_e):.6e} N/m")
        print(f"   Bending stiffness (12EI/L³) = {(12*E*I_y/l_e**3):.6e} N/m")

    # Calculate direction cosines (element orientation in global coordinates)
    l_x = (X_j - X_i) / l_e  # cos(local_x, global_X)
    m_x = (Y_j - Y_i) / l_e  # cos(local_x, global_Y)
    n_x = (Z_j - Z_i) / l_e  # cos(local_x, global_Z)

    # Create local y and z axes perpendicular to local x
    # Choose local y-axis to avoid singularity when element is vertical
    if abs(l_x) < 0.9:
        # Element is not too close to global X direction
        l_y = -m_x * n_x / np.sqrt(l_x**2 + m_x**2)
        m_y = l_x * n_x / np.sqrt(l_x**2 + m_x**2)
        n_y = -np.sqrt(l_x**2 + m_x**2)
    else:
        # Element is nearly aligned with global X direction
        l_y = 0
        m_y = 1
        n_y = 0
    
    # Normalize local y-axis
    mag_y = np.sqrt(l_y**2 + m_y**2 + n_y**2)
    l_y, m_y, n_y = l_y/mag_y, m_y/mag_y, n_y/mag_y
    
    # Local z-axis is cross product of local x and y
    l_z = m_x * n_y - n_x * m_y
    m_z = n_x * l_y - l_x * n_y
    n_z = l_x * m_y - m_x * l_y

    # Construct 3×3 transformation matrix from local to global
    T_small = np.array([
        [l_x, m_x, n_x],
        [l_y, m_y, n_y],
        [l_z, m_z, n_z]
    ])
    
    # Expand to 12×12 transformation matrix for beam element (6 DOF per node × 2 nodes)
    T = np.zeros((12, 12))
    T[0:3, 0:3] = T_small
    T[3:6, 3:6] = T_small
    T[6:9, 6:9] = T_small
    T[9:12, 9:12] = T_small

    # Define the 12×12 local stiffness matrix K_local for BEAM element
    # Includes: axial, bending (2 axes), torsion
    K_local = np.zeros((12, 12))
    
    # Axial stiffness terms (DOF 1 and 7: u1x, u2x)
    K_local[0, 0] = A * E / l_e
    K_local[0, 6] = -A * E / l_e
    K_local[6, 0] = -A * E / l_e
    K_local[6, 6] = A * E / l_e
    
    # Torsional stiffness terms (DOF 4 and 10: θ1x, θ2x)
    K_local[3, 3] = G * J / l_e
    K_local[3, 9] = -G * J / l_e
    K_local[9, 3] = -G * J / l_e
    K_local[9, 9] = G * J / l_e
    
    # Bending about local z-axis (in local xy-plane)
    # DOF 2, 5, 8, 11: u1y, θ1z, u2y, θ2z
    K_local[1, 1] = 12 * E * I_z / l_e**3
    K_local[1, 5] = 6 * E * I_z / l_e**2
    K_local[1, 7] = -12 * E * I_z / l_e**3
    K_local[1, 11] = 6 * E * I_z / l_e**2
    
    K_local[5, 1] = 6 * E * I_z / l_e**2
    K_local[5, 5] = 4 * E * I_z / l_e
    K_local[5, 7] = -6 * E * I_z / l_e**2
    K_local[5, 11] = 2 * E * I_z / l_e
    
    K_local[7, 1] = -12 * E * I_z / l_e**3
    K_local[7, 5] = -6 * E * I_z / l_e**2
    K_local[7, 7] = 12 * E * I_z / l_e**3
    K_local[7, 11] = -6 * E * I_z / l_e**2
    
    K_local[11, 1] = 6 * E * I_z / l_e**2
    K_local[11, 5] = 2 * E * I_z / l_e
    K_local[11, 7] = -6 * E * I_z / l_e**2
    K_local[11, 11] = 4 * E * I_z / l_e
    
    # Bending about local y-axis (in local xz-plane)
    # DOF 3, 6, 9, 12: u1z, θ1y, u2z, θ2y
    K_local[2, 2] = 12 * E * I_y / l_e**3
    K_local[2, 4] = -6 * E * I_y / l_e**2
    K_local[2, 8] = -12 * E * I_y / l_e**3
    K_local[2, 10] = -6 * E * I_y / l_e**2
    
    K_local[4, 2] = -6 * E * I_y / l_e**2
    K_local[4, 4] = 4 * E * I_y / l_e
    K_local[4, 8] = 6 * E * I_y / l_e**2
    K_local[4, 10] = 2 * E * I_y / l_e
    
    K_local[8, 2] = -12 * E * I_y / l_e**3
    K_local[8, 4] = 6 * E * I_y / l_e**2
    K_local[8, 8] = 12 * E * I_y / l_e**3
    K_local[8, 10] = 6 * E * I_y / l_e**2
    
    K_local[10, 2] = -6 * E * I_y / l_e**2
    K_local[10, 4] = 2 * E * I_y / l_e
    K_local[10, 8] = 6 * E * I_y / l_e**2
    K_local[10, 10] = 4 * E * I_y / l_e

    # Transform local stiffness to global coordinates
    K_global = T.T @ K_local @ T

    # Map local DOFs to global DOFs (6 DOF per node)
    global_dofs = [
        (node1_num - 1) * 6 + 0,  # Node 1 ux
        (node1_num - 1) * 6 + 1,  # Node 1 uy
        (node1_num - 1) * 6 + 2,  # Node 1 uz
        (node1_num - 1) * 6 + 3,  # Node 1 rx
        (node1_num - 1) * 6 + 4,  # Node 1 ry
        (node1_num - 1) * 6 + 5,  # Node 1 rz
        (node2_num - 1) * 6 + 0,  # Node 2 ux
        (node2_num - 1) * 6 + 1,  # Node 2 uy
        (node2_num - 1) * 6 + 2,  # Node 2 uz
        (node2_num - 1) * 6 + 3,  # Node 2 rx
        (node2_num - 1) * 6 + 4,  # Node 2 ry
        (node2_num - 1) * 6 + 5   # Node 2 rz
    ]

    # Add K_global values to global stiffness matrix
    for i, global_i in enumerate(global_dofs):
        for j, global_j in enumerate(global_dofs):
            stiffness_matrix.iloc[global_i, global_j] += K_global[i, j]

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











####################SOLVE FEM SYSTEM FOR DISPLACEMENTS##########################

print("\n" + "="*80)
print("SOLVING FEM SYSTEM: K * u = F")
print("="*80)

# Step 1: Convert forces DataFrame to numpy vector (6 DOF per node)
F_global = forces['Value (N or N·m)'].fillna(0).to_numpy()
print(f"\nStep 1: Force vector F created ({len(F_global)} DOFs)")
print(f"  - Total nodes: {num_nodes}")
print(f"  - DOF per node: 6 (3 forces + 3 moments)")

# Step 2: Identify known (fixed) and unknown (free) DOFs
known_dofs = []  # DOFs with prescribed displacement (u = 0)
unknown_dofs = []  # DOFs to be solved

for i in range(len(displacements)):
    if pd.notna(displacements.loc[i, 'Value (m or rad)']):
        known_dofs.append(i)
    else:
        unknown_dofs.append(i)

print(f"\nStep 2: DOF classification")
print(f"  Known DOFs (fixed supports): {len(known_dofs)}")
print(f"  Unknown DOFs (free): {len(unknown_dofs)}")

# Step 3: Extract reduced stiffness matrix K_ff (free-free)
K_full = stiffness_matrix.to_numpy()
K_reduced = K_full[np.ix_(unknown_dofs, unknown_dofs)]
F_reduced = F_global[unknown_dofs]

print(f"\nStep 3: Reduced system extracted")
print(f"  Reduced K size: {K_reduced.shape[0]} × {K_reduced.shape[1]}")
print(f"  Condition number: {np.linalg.cond(K_reduced):.2e}")

# Step 4: Solve reduced system K_reduced * u_unknown = F_reduced
print(f"\nStep 4: Solving linear system...")
try:
    u_unknown = np.linalg.solve(K_reduced, F_reduced)
    print(f"  ✓ System solved successfully")
except np.linalg.LinAlgError as e:
    print(f"  ✗ Error solving system: {e}")
    print(f"  Matrix may be singular or ill-conditioned")
    u_unknown = np.zeros(len(unknown_dofs))

# Step 5: Reconstruct full displacement vector
u_full = np.zeros(n_dof)
u_full[known_dofs] = 0.0  # Known displacements (supports)
u_full[unknown_dofs] = u_unknown  # Solved displacements

# Update displacements DataFrame (includes translations and rotations)
displacements['Value (m or rad)'] = u_full

print(f"\nStep 5: Full displacement vector reconstructed")
print(f"  Total DOFs: {len(u_full)}")
print(f"  Max displacement: {np.abs(u_full).max():.6e} m")
print(f"  Max displacement (mm): {np.abs(u_full).max() * 1000:.6f} mm")

# Display displacement results
print("\n" + "="*80)
print("DISPLACEMENT RESULTS")
print("="*80)

# Show displacements for each node (translations and rotations)
for node_num in range(1, num_nodes + 1):
    ux_idx = (node_num - 1) * 6 + 0
    uy_idx = (node_num - 1) * 6 + 1
    uz_idx = (node_num - 1) * 6 + 2
    rx_idx = (node_num - 1) * 6 + 3
    ry_idx = (node_num - 1) * 6 + 4
    rz_idx = (node_num - 1) * 6 + 5
    
    ux = u_full[ux_idx]
    uy = u_full[uy_idx]
    uz = u_full[uz_idx]
    rx = u_full[rx_idx]
    ry = u_full[ry_idx]
    rz = u_full[rz_idx]
    
    trans_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
    rot_magnitude = np.sqrt(rx**2 + ry**2 + rz**2)
    
    # Only print nodes with significant displacement (> 1 μm or > 1 μrad)
    if trans_magnitude > 1e-6 or rot_magnitude > 1e-6:
        print(f"Node {node_num:2d}: Ux={ux*1000:8.4f} mm, Uy={uy*1000:8.4f} mm, Uz={uz*1000:8.4f} mm, |U|={trans_magnitude*1000:8.4f} mm")
        print(f"          Rx={rx*1000:8.4f} mrad, Ry={ry*1000:8.4f} mrad, Rz={rz*1000:8.4f} mrad, |R|={rot_magnitude*1000:8.4f} mrad")

# Display displacements DataFrame in interactive window
display_matrix_table(displacements, "Solved Displacements")

# Step 6: Calculate reaction forces at supports
print("\n" + "="*80)
print("REACTION FORCES AT SUPPORTS")
print("="*80)

reactions = K_full[known_dofs, :] @ u_full

for i, dof_idx in enumerate(known_dofs):
    node_num = dof_idx // 6 + 1
    dof_type = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'][dof_idx % 6]
    reaction = reactions[i]
    unit = 'N' if dof_idx % 6 < 3 else 'N·m'
    print(f"Node {node_num} - {dof_type}: {reaction:10.2f} {unit}")

# Sum of reactions (should balance applied forces)
total_reaction = np.sum(reactions)
total_applied_force = np.sum(F_global)
print(f"\nForce balance check:")
print(f"  Total applied force: {total_applied_force:.2f} N")
print(f"  Total reaction force: {total_reaction:.2f} N")
print(f"  Difference: {abs(total_applied_force + total_reaction):.2e} N")

print("\n" + "="*80)





















####################Plot Undeformed and Deformed Structure Comparison##########################

# Create interactive Plotly 3D plot comparing undeformed and deformed structures
fig_deformed = go.Figure()

# Calculate deformed node positions (using translations only, rotations affect element orientation)
deformed_points = points.copy()
for node_num in range(1, num_nodes + 1):
    ux_idx = (node_num - 1) * 6 + 0  # Translation in X
    uy_idx = (node_num - 1) * 6 + 1  # Translation in Y
    uz_idx = (node_num - 1) * 6 + 2  # Translation in Z
    
    # Get node index in points DataFrame
    node_idx = node_num - 1
    
    # Add translational displacements to original coordinates
    # Note: Rotations (rx, ry, rz) affect element orientation but not node position
    deformed_points.loc[node_idx, 'X'] = points.iloc[node_idx]['X'] + u_full[ux_idx]
    deformed_points.loc[node_idx, 'Y'] = points.iloc[node_idx]['Y'] + u_full[uy_idx]
    deformed_points.loc[node_idx, 'Z'] = points.iloc[node_idx]['Z'] + u_full[uz_idx]

# Add undeformed elements (lines between nodes)
edge_x_undeformed = []
edge_y_undeformed = []
edge_z_undeformed = []

for idx in range(len(elements)):
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    # Get coordinates
    x1, y1, z1 = points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    edge_x_undeformed.extend([x1, x2, None])
    edge_y_undeformed.extend([y1, y2, None])
    edge_z_undeformed.extend([z1, z2, None])

# Add undeformed elements as lines
fig_deformed.add_trace(go.Scatter3d(
    x=edge_x_undeformed,
    y=edge_y_undeformed,
    z=edge_z_undeformed,
    mode='lines',
    line=dict(color='blue', width=2),
    hoverinfo='skip',
    name='Undeformed Elements'
))

# Add deformed elements (lines between deformed nodes)
edge_x_deformed = []
edge_y_deformed = []
edge_z_deformed = []

for idx in range(len(elements)):
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    # Get deformed coordinates
    x1, y1, z1 = deformed_points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = deformed_points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    edge_x_deformed.extend([x1, x2, None])
    edge_y_deformed.extend([y1, y2, None])
    edge_z_deformed.extend([z1, z2, None])

# Add deformed elements as lines
fig_deformed.add_trace(go.Scatter3d(
    x=edge_x_deformed,
    y=edge_y_deformed,
    z=edge_z_deformed,
    mode='lines',
    line=dict(color='red', width=3),
    hoverinfo='skip',
    name='Deformed Elements'
))

# Add undeformed nodes
fig_deformed.add_trace(go.Scatter3d(
    x=points['X'],
    y=points['Y'],
    z=points['Z'],
    mode='markers',
    marker=dict(size=6, color='blue', opacity=0.6),
    hovertemplate='<b>Undeformed Node %{text}</b><br>X: %{x:.6f} m<br>Y: %{y:.6f} m<br>Z: %{z:.6f} m<extra></extra>',
    text=[f"{int(n)}" for n in points['Node Number']],
    name='Undeformed Nodes'
))

# Add deformed nodes
fig_deformed.add_trace(go.Scatter3d(
    x=deformed_points['X'],
    y=deformed_points['Y'],
    z=deformed_points['Z'],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.9),
    hovertemplate='<b>Deformed Node %{text}</b><br>X: %{x:.6f} m<br>Y: %{y:.6f} m<br>Z: %{z:.6f} m<extra></extra>',
    text=[f"{int(n)}" for n in deformed_points['Node Number']],
    name='Deformed Nodes'
))

# Set equal aspect ratio and layout
fig_deformed.update_layout(
    title=f'Deformed vs Undeformed Structure<br><sub>Max Displacement: {np.abs(u_full).max()*1000:.4f} mm</sub>',
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=900,
    hovermode='closest',
    showlegend=True
)

fig_deformed.show()

print(f"\nDeformed structure visualization complete.")
print(f"  Blue nodes: Original (undeformed) positions")
print(f"  Red nodes: Deformed positions")
print(f"  Green dashed lines: Displacement vectors")










####################Plot MAGNIFIED Deformed Structure for Better Visualization##########################

# Create magnified deformation plot to better visualize small displacements
MAGNIFICATION_FACTOR = 1000  # Magnify displacements by 1000x for visibility

print(f"\n" + "="*80)
print(f"CREATING MAGNIFIED DEFORMATION PLOT (Scale Factor: {MAGNIFICATION_FACTOR}x)")
print("="*80)

fig_magnified = go.Figure()

# Calculate MAGNIFIED deformed node positions
magnified_points = points.copy()
for node_num in range(1, num_nodes + 1):
    ux_idx = (node_num - 1) * 3 + 0
    uy_idx = (node_num - 1) * 3 + 1
    uz_idx = (node_num - 1) * 3 + 2
    
    # Get node index in points DataFrame
    node_idx = node_num - 1
    
    # Add MAGNIFIED displacements to original coordinates
    magnified_points.loc[node_idx, 'X'] = points.iloc[node_idx]['X'] + u_full[ux_idx] * MAGNIFICATION_FACTOR
    magnified_points.loc[node_idx, 'Y'] = points.iloc[node_idx]['Y'] + u_full[uy_idx] * MAGNIFICATION_FACTOR
    magnified_points.loc[node_idx, 'Z'] = points.iloc[node_idx]['Z'] + u_full[uz_idx] * MAGNIFICATION_FACTOR

# Add undeformed elements
edge_x_undeformed_mag = []
edge_y_undeformed_mag = []
edge_z_undeformed_mag = []

for idx in range(len(elements)):
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    x1, y1, z1 = points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    edge_x_undeformed_mag.extend([x1, x2, None])
    edge_y_undeformed_mag.extend([y1, y2, None])
    edge_z_undeformed_mag.extend([z1, z2, None])

fig_magnified.add_trace(go.Scatter3d(
    x=edge_x_undeformed_mag,
    y=edge_y_undeformed_mag,
    z=edge_z_undeformed_mag,
    mode='lines',
    line=dict(color='cyan', width=4),
    hoverinfo='skip',
    name='Undeformed',
    opacity=0.9
))

# Add MAGNIFIED deformed elements
edge_x_magnified = []
edge_y_magnified = []
edge_z_magnified = []

for idx in range(len(elements)):
    node1_idx = int(elements.iloc[idx]['Node1']) - 1
    node2_idx = int(elements.iloc[idx]['Node2']) - 1
    
    x1, y1, z1 = magnified_points.iloc[node1_idx][['X', 'Y', 'Z']]
    x2, y2, z2 = magnified_points.iloc[node2_idx][['X', 'Y', 'Z']]
    
    edge_x_magnified.extend([x1, x2, None])
    edge_y_magnified.extend([y1, y2, None])
    edge_z_magnified.extend([z1, z2, None])

fig_magnified.add_trace(go.Scatter3d(
    x=edge_x_magnified,
    y=edge_y_magnified,
    z=edge_z_magnified,
    mode='lines',
    line=dict(color='red', width=4),
    hoverinfo='skip',
    name='Deformed (Magnified)'
))

# Add undeformed nodes
fig_magnified.add_trace(go.Scatter3d(
    x=points['X'],
    y=points['Y'],
    z=points['Z'],
    mode='markers',
    marker=dict(size=8, color='cyan', opacity=0.9),
    hoverinfo='skip',
    name='Undeformed Nodes'
))

# Add MAGNIFIED deformed nodes
fig_magnified.add_trace(go.Scatter3d(
    x=magnified_points['X'],
    y=magnified_points['Y'],
    z=magnified_points['Z'],
    mode='markers+text',
    marker=dict(size=6, color='red', opacity=0.9),
    text=[f"{int(n)}" for n in magnified_points['Node Number']],
    textposition='top center',
    textfont=dict(size=8),
    hovertemplate='<b>Node %{text}</b><br>Magnified X: %{x:.3f} m<br>Magnified Y: %{y:.3f} m<br>Magnified Z: %{z:.3f} m<extra></extra>',
    name='Deformed Nodes'
))

# Set equal aspect ratio and layout
fig_magnified.update_layout(
    title=f'MAGNIFIED Deformed Structure ({MAGNIFICATION_FACTOR}x)<br><sub>Actual Max Displacement: {np.abs(u_full).max()*1000:.4f} mm</sub>',
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1200,
    height=900,
    hovermode='closest',
    showlegend=True
)

fig_magnified.show()

print(f"\nMagnified deformation plot created successfully!")
print(f"  Magnification factor: {MAGNIFICATION_FACTOR}x")
print(f"  Actual max displacement: {np.abs(u_full).max()*1000:.4f} mm")
print(f"  Apparent displacement in plot: {np.abs(u_full).max()*MAGNIFICATION_FACTOR*1000:.2f} mm")
print(f"  Note: This is for VISUALIZATION only - actual displacements are {MAGNIFICATION_FACTOR}x smaller!")
print("="*80)








# Get node coordinates
node1 = points[points['Node Number'] == node1_num].iloc[0]
node2 = points[points['Node Number'] == node2_num].iloc[0]

X_i = node1['X']
Y_i = node1['Y']
Z_i = node1['Z']

X_j = node2['X']
Y_j = node2['Y']
Z_j = node2['Z']

# ========== DEBUG: Check element length units ==========
# Calculate ACTUAL element length from coordinates
dx = X_j - X_i
dy = Y_j - Y_i
dz = Z_j - Z_i
l_actual = np.sqrt(dx**2 + dy**2 + dz**2)

# Get stored length from DataFrame
l_e = elem['Element Length']

# Compare stored vs actual length
length_error = abs(l_actual - l_e)
relative_error = length_error / l_actual if l_actual > 0 else 0

if element_idx < 5 or relative_error > 0.01:  # Print first 5 elements or any with >1% error
    print(f"\n🔍 Element {element_idx+1} (Nodes {node1_num}-{node2_num}):")
    print(f"   Stored length (l_e): {l_e:.6f} m")
    print(f"   Actual length:       {l_actual:.6f} m")
    print(f"   Difference:          {length_error:.2e} m ({relative_error*100:.2f}%)")
    print(f"   Coordinates: ({X_i:.3f}, {Y_i:.3f}, {Z_i:.3f}) → ({X_j:.3f}, {Y_j:.3f}, {Z_j:.3f})")

# Use ACTUAL length for direction cosines (critical fix!)
l_e_corrected = l_actual
# =======================================================

# Calculate direction cosines using CORRECTED length
l_y = dx / l_e_corrected  # cos(x, X)
m_y = dy / l_e_corrected  # cos(x, Y)
n_y = dz / l_e_corrected  # cos(x, Z)

# Verify direction cosines sum to 1 (sanity check)
cos_sum = l_y**2 + m_y**2 + n_y**2
if abs(cos_sum - 1.0) > 1e-6:
    print(f"   ⚠️  WARNING: Direction cosines don't sum to 1.0: {cos_sum:.10f}")

# Define the local stiffness matrix K_e for the element
# Use l_e_corrected for stiffness calculation too!
K_e = (A * E / l_e_corrected) * np.array([
    [l_y**2,        l_y*m_y,      l_y*n_y,     -l_y**2,       -l_y*m_y,     -l_y*n_y    ],
    [l_y*m_y,       m_y**2,       m_y*n_y,     -l_y*m_y,      -m_y**2,      -m_y*n_y    ],
    [l_y*n_y,       m_y*n_y,      n_y**2,      -l_y*n_y,      -m_y*n_y,     -n_y**2     ],
    [-l_y**2,       -l_y*m_y,     -l_y*n_y,     l_y**2,        l_y*m_y,      l_y*n_y    ],
    [-l_y*m_y,      -m_y**2,      -m_y*n_y,     l_y*m_y,       m_y**2,       m_y*n_y    ],
    [-l_y*n_y,      -m_y*n_y,     -n_y**2,      l_y*n_y,       m_y*n_y,      n_y**2     ]
])