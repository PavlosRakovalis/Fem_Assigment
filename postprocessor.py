"""
FEM Post-Processor Module
=========================
This module reads FEM results from a .res file and creates visualizations
of the deformed structure, stresses, and other results.

Author: Copilot
Date: November 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os


# Configuration
BROWSER = 'chrome'
if BROWSER:
    pio.renderers.default = 'browser'


class FEMPostProcessor:
    """Handles visualization of FEM results"""
    
    def __init__(self):
        self.nodes = pd.DataFrame()
        self.elements = pd.DataFrame()
        self.displacements = pd.DataFrame()
        self.reactions = pd.DataFrame()
        self.element_forces = pd.DataFrame()
        self.summary = {}
        
    def read_input_file(self, input_filename='data/structure.dat'):
        """Read original structure data"""
        print("="*80)
        print("READING STRUCTURE DATA")
        print("="*80)
        
        if not os.path.exists(input_filename):
            raise FileNotFoundError(f"Input file not found: {input_filename}")
        
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        nodes_data = []
        elements_data = []
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line == 'NODES':
                current_section = 'NODES'
                continue
            elif line == 'ELEMENTS':
                current_section = 'ELEMENTS'
                continue
            elif line in ['BOUNDARY_CONDITIONS', 'LOADS', 'PARAMETERS', 'MATERIALS', 'END']:
                current_section = None
                continue
            
            if current_section == 'NODES':
                parts = line.split()
                if len(parts) == 4:
                    nodes_data.append({
                        'Node Number': int(parts[0]),
                        'X': float(parts[1]),
                        'Y': float(parts[2]),
                        'Z': float(parts[3])
                    })
            elif current_section == 'ELEMENTS':
                parts = line.split()
                if len(parts) == 6:
                    elements_data.append({
                        'Element Number': int(parts[0]),
                        'Node1': int(parts[1]),
                        'Node2': int(parts[2])
                    })
        
        self.nodes = pd.DataFrame(nodes_data)
        self.elements = pd.DataFrame(elements_data)
        
        print(f"✓ Structure data read: {len(self.nodes)} nodes, {len(self.elements)} elements")
        
    def read_results_file(self, results_filename='data/structure.res'):
        """Read FEM results from .res file"""
        print("\nREADING RESULTS FILE")
        print("="*80)
        
        if not os.path.exists(results_filename):
            raise FileNotFoundError(f"Results file not found: {results_filename}")
        
        with open(results_filename, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        displacements_data = []
        reactions_data = []
        element_forces_data = []
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line == 'DISPLACEMENTS':
                current_section = 'DISPLACEMENTS'
                continue
            elif line == 'REACTIONS':
                current_section = 'REACTIONS'
                continue
            elif line == 'ELEMENT_FORCES':
                current_section = 'ELEMENT_FORCES'
                continue
            elif line == 'SUMMARY':
                current_section = 'SUMMARY'
                continue
            elif line == 'END':
                break
            
            if current_section == 'DISPLACEMENTS':
                parts = line.split()
                if len(parts) == 5:
                    displacements_data.append({
                        'Node Number': int(parts[0]),
                        'Ux': float(parts[1]),
                        'Uy': float(parts[2]),
                        'Uz': float(parts[3]),
                        'Magnitude': float(parts[4])
                    })
            elif current_section == 'REACTIONS':
                parts = line.split()
                if len(parts) == 4:
                    reactions_data.append({
                        'Node Number': int(parts[0]),
                        'Rx': float(parts[1]),
                        'Ry': float(parts[2]),
                        'Rz': float(parts[3])
                    })
            elif current_section == 'ELEMENT_FORCES':
                parts = line.split()
                if len(parts) == 4:
                    element_forces_data.append({
                        'Element Number': int(parts[0]),
                        'Axial Force': float(parts[1]),
                        'Stress': float(parts[2]),
                        'Strain': float(parts[3])
                    })
            elif current_section == 'SUMMARY':
                parts = line.split()
                if len(parts) == 2:
                    self.summary[parts[0]] = float(parts[1])
        
        self.displacements = pd.DataFrame(displacements_data)
        self.reactions = pd.DataFrame(reactions_data)
        self.element_forces = pd.DataFrame(element_forces_data)
        
        print(f"✓ Results read successfully: {results_filename}")
        print(f"  - {len(self.displacements)} node displacements")
        print(f"  - {len(self.reactions)} reactions")
        print(f"  - {len(self.element_forces)} element results")
        
    def plot_deformed_structure(self, magnification=1.0, show_plot=True):
        """Plot undeformed and deformed structure"""
        print("\nCREATING DEFORMATION PLOT")
        print("="*80)
        
        fig = go.Figure()
        
        # Undeformed structure
        edge_x_undef, edge_y_undef, edge_z_undef = [], [], []
        for idx in range(len(self.elements)):
            node1_num = int(self.elements.iloc[idx]['Node1'])
            node2_num = int(self.elements.iloc[idx]['Node2'])
            
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            edge_x_undef.extend([node1['X'], node2['X'], None])
            edge_y_undef.extend([node1['Y'], node2['Y'], None])
            edge_z_undef.extend([node1['Z'], node2['Z'], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x_undef, y=edge_y_undef, z=edge_z_undef,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Undeformed'
        ))
        
        # Deformed structure
        deformed_nodes = self.nodes.copy()
        for idx in range(len(self.nodes)):
            node_num = int(self.nodes.iloc[idx]['Node Number'])
            disp = self.displacements[self.displacements['Node Number'] == node_num].iloc[0]
            
            deformed_nodes.loc[idx, 'X'] += disp['Ux'] * magnification
            deformed_nodes.loc[idx, 'Y'] += disp['Uy'] * magnification
            deformed_nodes.loc[idx, 'Z'] += disp['Uz'] * magnification
        
        edge_x_def, edge_y_def, edge_z_def = [], [], []
        for idx in range(len(self.elements)):
            node1_num = int(self.elements.iloc[idx]['Node1'])
            node2_num = int(self.elements.iloc[idx]['Node2'])
            
            node1 = deformed_nodes[deformed_nodes['Node Number'] == node1_num].iloc[0]
            node2 = deformed_nodes[deformed_nodes['Node Number'] == node2_num].iloc[0]
            
            edge_x_def.extend([node1['X'], node2['X'], None])
            edge_y_def.extend([node1['Y'], node2['Y'], None])
            edge_z_def.extend([node1['Z'], node2['Z'], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x_def, y=edge_y_def, z=edge_z_def,
            mode='lines',
            line=dict(color='red', width=3),
            name='Deformed'
        ))
        
        # Undeformed nodes
        fig.add_trace(go.Scatter3d(
            x=self.nodes['X'],
            y=self.nodes['Y'],
            z=self.nodes['Z'],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            name='Undeformed Nodes'
        ))
        
        # Deformed nodes
        fig.add_trace(go.Scatter3d(
            x=deformed_nodes['X'],
            y=deformed_nodes['Y'],
            z=deformed_nodes['Z'],
            mode='markers',
            marker=dict(size=6, color='red', opacity=0.9),
            name='Deformed Nodes'
        ))
        
        max_disp = self.summary.get('Max_Displacement', 0)
        title = f'Deformed Structure (Mag: {magnification}x)<br>'
        title += f'<sub>Max Displacement: {max_disp*1000:.4f} mm</sub>'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1200,
            height=900
        )
        
        if show_plot:
            fig.show()
        
        filename = f'deformed_structure_mag{magnification}.html'
        fig.write_html(filename)
        print(f"✓ Deformation plot saved to: {filename}")
        
        return fig
    
    def plot_magnified_deformation(self, magnification=1000, show_plot=True):
        """Plot highly magnified deformation"""
        print("\nCREATING MAGNIFIED DEFORMATION PLOT")
        print("="*80)
        
        fig = go.Figure()
        
        # Deformed nodes
        deformed_nodes = self.nodes.copy()
        for idx in range(len(self.nodes)):
            node_num = int(self.nodes.iloc[idx]['Node Number'])
            disp = self.displacements[self.displacements['Node Number'] == node_num].iloc[0]
            
            deformed_nodes.loc[idx, 'X'] += disp['Ux'] * magnification
            deformed_nodes.loc[idx, 'Y'] += disp['Uy'] * magnification
            deformed_nodes.loc[idx, 'Z'] += disp['Uz'] * magnification
        
        # Undeformed elements
        edge_x_undef, edge_y_undef, edge_z_undef = [], [], []
        for idx in range(len(self.elements)):
            node1_num = int(self.elements.iloc[idx]['Node1'])
            node2_num = int(self.elements.iloc[idx]['Node2'])
            
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            edge_x_undef.extend([node1['X'], node2['X'], None])
            edge_y_undef.extend([node1['Y'], node2['Y'], None])
            edge_z_undef.extend([node1['Z'], node2['Z'], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x_undef, y=edge_y_undef, z=edge_z_undef,
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Undeformed',
            opacity=0.9
        ))
        
        # Deformed elements
        edge_x_def, edge_y_def, edge_z_def = [], [], []
        for idx in range(len(self.elements)):
            node1_num = int(self.elements.iloc[idx]['Node1'])
            node2_num = int(self.elements.iloc[idx]['Node2'])
            
            node1 = deformed_nodes[deformed_nodes['Node Number'] == node1_num].iloc[0]
            node2 = deformed_nodes[deformed_nodes['Node Number'] == node2_num].iloc[0]
            
            edge_x_def.extend([node1['X'], node2['X'], None])
            edge_y_def.extend([node1['Y'], node2['Y'], None])
            edge_z_def.extend([node1['Z'], node2['Z'], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x_def, y=edge_y_def, z=edge_z_def,
            mode='lines',
            line=dict(color='red', width=4),
            name='Deformed (Magnified)'
        ))
        
        # Undeformed nodes
        fig.add_trace(go.Scatter3d(
            x=self.nodes['X'],
            y=self.nodes['Y'],
            z=self.nodes['Z'],
            mode='markers',
            marker=dict(size=8, color='cyan', opacity=0.9),
            name='Undeformed Nodes'
        ))
        
        # Deformed nodes
        fig.add_trace(go.Scatter3d(
            x=deformed_nodes['X'],
            y=deformed_nodes['Y'],
            z=deformed_nodes['Z'],
            mode='markers+text',
            marker=dict(size=6, color='red', opacity=0.9),
            text=[f"{int(n)}" for n in deformed_nodes['Node Number']],
            textposition='top center',
            textfont=dict(size=8),
            name='Deformed Nodes'
        ))
        
        max_disp = self.summary.get('Max_Displacement', 0)
        title = f'MAGNIFIED Deformed Structure ({magnification}x)<br>'
        title += f'<sub>Actual Max Displacement: {max_disp*1000:.4f} mm</sub>'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200,
            height=900
        )
        
        if show_plot:
            fig.show()
        
        filename = 'magnified_deformation.html'
        fig.write_html(filename)
        print(f"✓ Magnified deformation plot saved to: {filename}")
        
        return fig
    
    def plot_stress_distribution(self, show_plot=True):
        """Plot stress distribution in elements"""
        print("\nCREATING STRESS DISTRIBUTION PLOT")
        print("="*80)
        
        fig = go.Figure()
        
        # Get stress values
        stresses = self.element_forces['Stress'].values
        max_stress = np.abs(stresses).max()
        
        # Collect all element line segments with their stresses
        edge_x, edge_y, edge_z = [], [], []
        edge_stresses = []
        
        for idx in range(len(self.elements)):
            elem = self.elements.iloc[idx]
            node1_num = int(elem['Node1'])
            node2_num = int(elem['Node2'])
            
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            stress = self.element_forces.iloc[idx]['Stress']
            
            # Add line segment (use None to separate lines)
            edge_x.extend([node1['X'], node2['X'], None])
            edge_y.extend([node1['Y'], node2['Y'], None])
            edge_z.extend([node1['Z'], node2['Z'], None])
            # Repeat stress value for both endpoints and None
            edge_stresses.extend([stress, stress, stress])
        
        # Plot all elements with colorbar
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color=edge_stresses,
                width=8,  # Increased from 6 for more visibility
                colorscale='RdBu',  # Red=negative (compression), Blue=positive (tension) - REVERSED
                cmin=-max_stress,
                cmax=max_stress,
                colorbar=dict(
                    title=dict(
                        text="Stress (Pa)",
                        side="right"
                    ),
                    thickness=20,
                    len=0.7,
                    tickformat=".2e",
                    x=1.02
                )
            ),
            showlegend=False,
            hovertemplate='Stress: %{line.color:.2e} Pa<extra></extra>'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=self.nodes['X'],
            y=self.nodes['Y'],
            z=self.nodes['Z'],
            mode='markers',
            marker=dict(size=4, color='black'),
            name='Nodes',
            showlegend=True
        ))
        
        title = f'Element Stress Distribution<br>'
        title += f'<sub>Max Stress: {max_stress/1e6:.2f} MPa (Blue=Tension, Red=Compression)</sub>'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1200,
            height=900
        )
        
        if show_plot:
            fig.show()
        
        filename = 'stress_distribution.html'
        fig.write_html(filename)
        print(f"✓ Stress distribution plot saved to: {filename}")
        
        return fig
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        for key, value in self.summary.items():
            print(f"  {key.replace('_', ' ')}: {value:.6e}")
        
        print("\n  Top 5 Nodes by Displacement:")
        top_disps = self.displacements.nlargest(5, 'Magnitude')
        for idx in range(len(top_disps)):
            row = top_disps.iloc[idx]
            print(f"    Node {int(row['Node Number'])}: {row['Magnitude']*1000:.4f} mm")
        
        print("\n  Top 5 Elements by Stress:")
        top_stress = self.element_forces.copy()
        top_stress['Abs Stress'] = top_stress['Stress'].abs()
        top_stress = top_stress.nlargest(5, 'Abs Stress')
        for idx in range(len(top_stress)):
            row = top_stress.iloc[idx]
            print(f"    Element {int(row['Element Number'])}: {row['Stress']/1e6:.2f} MPa")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FEM POST-PROCESSOR")
    print("="*80 + "\n")
    
    # Create post-processor
    postprocessor = FEMPostProcessor()
    
    # Read data
    postprocessor.read_input_file('data/structure.dat')
    postprocessor.read_results_file('data/structure.res')
    
    # Print summary
    postprocessor.print_summary()
    
    # Create visualizations
    postprocessor.plot_deformed_structure(magnification=1.0, show_plot=True)
    postprocessor.plot_magnified_deformation(magnification=1000, show_plot=True)
    postprocessor.plot_stress_distribution(show_plot=True)
    
    print("\n" + "="*80)
    print("POST-PROCESSOR COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    print("Visualization files created:")
    print("  - deformed_structure_mag1.0.html")
    print("  - magnified_deformation.html")
    print("  - stress_distribution.html")


if __name__ == "__main__":
    main()
