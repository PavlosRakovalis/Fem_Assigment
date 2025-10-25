import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from geometry_utils import rotate_points_around_y









##### Define 2D points and plot them 

# Creating a DataFrame with node numbers and coordinates
points = pd.DataFrame({
    'Node Number': [1, 2, 3, 4],
    'X': [0.0, 1.0, 1.0, 0.0],
    'Y': [0.0, 0.0, 1.0, 1.0],
    'Z': [0.0, 0.0, 0.0, 0.0]
})





##I will now try to enter the nodes of the structure in a fast way using a a loop


L = 1.5 * 1.13
A = 1.2 * 1.69
phi = 60 + 9.1 #degrees


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









# Optionally rotate the structure about Y before plotting/analysis
# Set angle_deg to your desired rotation (degrees). Leave 0 for no rotation.
angle_deg = - phi # e.g., 30.0 to rotate 30 degrees about +Y
# You can also change the pivot point if needed: origin=(x0, y0, z0)
points = rotate_points_around_y(points, angle_deg, origin=(A, 0.0, 0.0))

# Convert to numpy array for compatibility with existing plotting code
points_array = points[['X', 'Y', 'Z']].to_numpy()

# Create interactive 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points['X'], points['Y'], points['Z'], color='blue', marker='o', s=100)

# Add node numbers as labels
for i, row in points.iterrows():
    ax.text(row['X'], row['Y'], row['Z'], 
            f'  Node {row["Node Number"]}', 
            fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points Plot')
ax.grid(True)

plt.show()








##### Define elements connecting the points and plot them



elements = pd.DataFrame({
    'Element Number': [1, 2, 3, 4, 5],
    'Node1': [1, 2, 3, 4, 1],
    'Node2': [2, 3, 4, 1, 3],
    'E': [0, 0, 0, 0, 0],
    'V': [0, 0, 0, 0, 0]
})





# Create interactive 3D plot for elements
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot points
ax2.scatter(points['X'], points['Y'], points['Z'], color='blue', marker='o', s=100)

# Add node numbers as labels
for i, row in points.iterrows():
    ax2.text(row['X'], row['Y'], row['Z'], 
            f'  Node {row["Node Number"]}', 
            fontsize=10)

# Plot the connections between points
for i in range(len(elements)):
    n1 = elements['Node1'][i] - 1  # Adjust for 0-based indexing
    n2 = elements['Node2'][i] - 1
    ax2.plot([points['X'].iloc[n1], points['X'].iloc[n2]], 
             [points['Y'].iloc[n1], points['Y'].iloc[n2]],
             [points['Z'].iloc[n1], points['Z'].iloc[n2]], 'k-', linewidth=2)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Elements Plot')
ax2.grid(True)

plt.show()
























# Create empty stiffness matrix (3 DOF per node: x, y, z)
num_nodes = len(points)
n_dof = 3 * num_nodes
row_labels = [f"F{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
col_labels = [f"V{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
stiffness_matrix = pd.DataFrame(np.zeros((n_dof, n_dof)), index=row_labels, columns=col_labels)






# For every element number "a" the stiffeness matrix in its local coordinates is equal with the image that i gave you. This matrix is the k_e_a.
# I want you to 

# Loop through each element in the elements DataFrame
for i in range(len(elements)):
    # Get node coordinates for this element
    node1_idx = elements['Node1'][i] - 1
    node2_idx = elements['Node2'][i] - 1

    x1, y1, z1 = points.loc[node1_idx, ['X', 'Y', 'Z']]
    x2, y2, z2 = points.loc[node2_idx, ['X', 'Y', 'Z']]

    # Calculate element length
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    print(f"Element {elements['Element Number'][i]}: Length = {length:.4f}")

    # Compute local stiffness matrix k_e_a for this element 
    E = elements['E'][i]
    A = 1.0  # Assume unit cross-sectional area for simplicity  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!prosoxi!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Den ishiei prpeei na to allaskw avrio 




# Display the stiffness matrix in an interactive scrollable window using tkinter
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

# Display the stiffness matrix
display_matrix_table(stiffness_matrix, "Stiffness Matrix")



