# Conversion from Bar Elements to Beam Elements

## Overview
The FEM code has been successfully converted from simple bar/truss elements (axial forces only) to beam elements (axial forces + bending moments + shear forces). This enables the analysis of structures under bending loads.

## Key Changes

### 1. **Degrees of Freedom (DOF)**
- **Before**: 3 DOF per node (ux, uy, uz) for 3D truss
- **After**: 3 DOF per node (ux, uy, rotation_z) for 2D planar frame
- Each beam element has 6 DOF total (3 per node)

### 2. **Element Stiffness Matrix**
The element stiffness matrix has been updated to include both axial and bending terms:

**Axial terms** (unchanged):
- k_axial = E * A / L

**Bending terms** (NEW - Euler-Bernoulli beam):
- k_bending includes shear and moment contributions
- Uses moment of inertia (I) for bending stiffness
- 6×6 local stiffness matrix instead of 2×2

### 3. **Transformation Matrix**
Updated from 2×6 (truss) to 6×6 (beam):
```
T = [c  s  0  0  0  0]
    [-s c  0  0  0  0]
    [0  0  1  0  0  0]
    [0  0  0  c  s  0]
    [0  0  0 -s  c  0]
    [0  0  0  0  0  1]
```
Where c = cos(θ), s = sin(θ)

### 4. **Material Properties**
Added **Moment of Inertia (I)** for each element:
- Calculated from cross-sectional area
- For square section: I = (h^4) / 12 where h = sqrt(A)
- Required for bending stiffness calculations

### 5. **Boundary Conditions**
- **Before**: {Ux, Uy, Uz} with 0=free, 1=fixed
- **After**: {Ux, Uy, Rz} with 0=free, 1=fixed
- Rz controls rotational constraint about Z-axis

### 6. **Loads**
- **Before**: {Fx, Fy, Fz} - forces only
- **After**: {Fx, Fy, Mz} - forces and moments
- Mz is the applied moment about Z-axis (N·m)

### 7. **Element Forces/Results**
Beam elements now calculate:
- **Axial Force** (N)
- **Shear Force** (N) - NEW
- **Bending Moments** at both ends (N·m) - NEW
- **Axial Stress** (Pa)
- **Bending Stress** (Pa) - NEW
- **Combined Stress** (Pa) - NEW (axial + bending)
- **Axial Strain**

### 8. **File Format Changes**

**structure.dat** (input):
```
ELEMENTS
# Elem_ID Node1 Node2 CrossSection MomentOfInertia E nu

BOUNDARY_CONDITIONS
# Node_ID Ux Uy Rz (0=free, 1=fixed)

LOADS
# Node_ID Fx Fy Mz (N, N, N·m)
```

**structure.res** (output):
```
DISPLACEMENTS
# Node_ID Ux(m) Uy(m) Rz(rad) Magnitude(m)

REACTIONS
# Node_ID Rx(N) Ry(N) Mrz(N·m)

ELEMENT_FORCES
# Elem_ID Axial_Force(N) Shear_Force(N) Moment1(N·m) Moment2(N·m) 
#         Axial_Stress(Pa) Bending_Stress(Pa) Combined_Stress(Pa) Strain
```

## Modified Files

1. **solver.py**
   - Updated stiffness matrix assembly for beam elements
   - Modified force vector handling for moments
   - Updated boundary condition application
   - Enhanced element force calculations
   - Modified results export format

2. **preprocessor.py**
   - Added moment of inertia calculations
   - Updated boundary condition method signature
   - Modified load method to handle moments
   - Updated export format

3. **postprocessor.py**
   - Updated results file reading for new format
   - Modified stress visualization to use combined stress
   - Updated displacement visualization (2D planar)

4. **transformation.py**
   - (Transformation now handled directly in solver)

## Theory: Beam Element Formulation

### Local Stiffness Matrix (6×6)
For a 2D beam element in local coordinates:
- Rows/Cols 1,4: Axial DOF
- Rows/Cols 2,3,5,6: Transverse deflection and rotation DOF

### Coordinate Systems
- **Local**: Aligned with element axis (x along element)
- **Global**: Structure coordinate system (X-Y plane)
- **Transformation**: K_global = T^T * K_local * T

### Bending Stiffness Terms
Based on Euler-Bernoulli beam theory:
- Assumes plane sections remain plane
- Neglects shear deformation
- Valid for slender beams (L/h > 10)

## Usage Example

```python
from preprocessor import FEMPreProcessor

# Create preprocessor
pre = FEMPreProcessor()

# Define boundary conditions (with rotation constraint)
pre.add_boundary_condition(1, ux=1, uy=1, rz=1)  # Fixed support

# Apply loads (force and moment)
pre.add_load(5, fx=1000, fy=500, mz=100)  # Force + Moment

# Export and solve as before
pre.export_to_file('data/structure.dat')
```

## Benefits of Beam Elements

1. **More Realistic**: Captures bending behavior of structural members
2. **Bending Loads**: Can now analyze frames, beams, portals
3. **Moments**: Can apply and calculate moments
4. **Combined Stress**: Accounts for both axial and bending stresses
5. **Rotational Constraints**: Can model moment connections and hinges

## Validation

The conversion maintains:
- ✅ Equilibrium (forces and moments balance)
- ✅ Compatibility (displacements continuous)
- ✅ Material behavior (linear elastic)
- ✅ Geometric accuracy

## Notes

- The implementation uses 2D planar frame theory (XY plane)
- For 3D space frames, extend to 6 DOF per node (ux, uy, uz, rx, ry, rz)
- Shear deformation can be added using Timoshenko beam theory
- Large deformations would require geometric nonlinearity
