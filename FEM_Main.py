import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk



##### Define 2D points and plot them 

# Creating a DataFrame with node numbers and coordinates
points = pd.DataFrame({
    'Node Number': [1, 2, 3, 4],
    'X': [0.0, 1.0, 1.0, 0.0],
    'Y': [0.0, 0.0, 1.0, 1.0],
    'Z': [0.0, 0.0, 0.0, 0.0]
})

# Convert to numpy array for compatibility with existing plotting code
points_array = points[['X', 'Y', 'Z']].to_numpy()

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(points['X'], points['Y'], color='blue', marker='o')

# Add node numbers as labels
for i, row in points.iterrows():
    plt.annotate(f'Node {row["Node Number"]}', 
                (row['X'], row['Y']), 
                xytext=(5, 5), 
                textcoords='offset points')

plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Points Plot')
plt.axis('equal')
plt.show()








##### Define elements connecting the points and plot them



elements = pd.DataFrame({
    'Node1': [1, 2, 3, 4, 1],
    'Node2': [2, 3, 4, 1, 3],
    'E': [0, 0, 0, 0, 0],
    'V': [0, 0, 0, 0, 0]
})

# Plot the connections between points
for i in range(len(elements)):
    n1 = elements['Node1'][i] - 1  # Adjust for 0-based indexing
    n2 = elements['Node2'][i] - 1
    plt.plot([points['X'].iloc[n1], points['X'].iloc[n2]], 
             [points['Y'].iloc[n1], points['Y'].iloc[n2]], 'k-')

plt.show()




# Create empty stiffness matrix (3 DOF per node: x, y, z)
num_nodes = len(points)
n_dof = 3 * num_nodes
row_labels = [f"F{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
col_labels = [f"V{n}{axis}" for n in range(1, num_nodes + 1) for axis in ("x", "y", "z")]
stiffness_matrix = pd.DataFrame(np.zeros((n_dof, n_dof)), index=row_labels, columns=col_labels)











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