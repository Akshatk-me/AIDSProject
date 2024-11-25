from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Define grid dimensions and properties
nx, ny = 50, 50
Lx, Ly = 2.0, 2.0
dx = Lx / nx
dy = Ly / ny

# Create the mesh
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Define fluid properties
viscosity = 0.1  # Kinematic viscosity
density = 1.0    # Fluid density

# Initialize variables
velocity_x = CellVariable(mesh=mesh, name="Velocity X", value=0.0)
velocity_y = CellVariable(mesh=mesh, name="Velocity Y", value=0.0)
pressure = CellVariable(mesh=mesh, name="Pressure", value=0.0)

# Define the equations
velocity_x_eq = (
    TransientTerm(coeff=density)
    == DiffusionTerm(coeff=viscosity)
    - pressure.grad[0]
)

velocity_y_eq = (
    TransientTerm(coeff=density)
    == DiffusionTerm(coeff=viscosity)
    - pressure.grad[1]
)

pressure_eq = DiffusionTerm(coeff=1.0) == -(
    velocity_x.grad[0] + velocity_y.grad[1]
)

# Set boundary conditions
velocity_x.constrain(0, mesh.facesLeft)  # No-slip condition on left wall
velocity_x.constrain(1, mesh.facesRight) # Constant velocity on right wall
velocity_y.constrain(0, mesh.exteriorFaces)  # No normal flow at walls
pressure.constrain(0, mesh.facesRight)      # Pressure reference point

# Simulation parameters
time_step = 0.01
steps = 200
frames_to_save = 100

# Initialize storage for animation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
velocity_field = np.zeros((frames_to_save, ny, nx))  # Store data for animation

# Simulate and store frames
for step in range(steps):
    velocity_x_eq.solve(var=velocity_x, dt=time_step)
    velocity_y_eq.solve(var=velocity_y, dt=time_step)
    pressure_eq.solve(var=pressure)

    if step < frames_to_save:
        # Update velocity magnitude for animation
        velocity_field[step, :, :] = np.sqrt(velocity_x.value.reshape((ny, nx))**2 +
                                             velocity_y.value.reshape((ny, nx))**2)

# Create animation
fig, ax = plt.subplots(figsize=(6, 6))
contour = ax.contourf(X, Y, velocity_field[0], cmap="viridis", levels=50)
plt.colorbar(contour, ax=ax)

def update(frame):
    """Update the contour plot for each frame."""
    ax.clear()
    ax.set_title(f"Time Step: {frame}")
    ax.contourf(X, Y, velocity_field[frame], cmap="viridis", levels=50)

ani = FuncAnimation(fig, update, frames=frames_to_save, interval=50)
plt.show()

