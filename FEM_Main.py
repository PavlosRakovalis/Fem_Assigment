import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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





