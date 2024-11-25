import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx, ny = 50, 50  # Grid points
lx, ly = 2.0, 2.0  # Domain size (m)
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
nu = 0.1  # Kinematic viscosity (m^2/s)
dt = 0.001  # Time step (s)
nt = 500  # Number of time steps
rho = 1.0  # Density (kg/m^3)

# Initialize fields
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # Pressure
b = np.zeros((ny, nx))  # Right-hand side of the Poisson equation

# Boundary conditions
def apply_boundary_conditions(u, v):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 1  # Lid-driven cavity top velocity

    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

# Build the Poisson equation RHS
def build_b(u, v, rho, dt, dx, dy):
    b = (rho * (1 / dt * 
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + 
                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

# Solve the Poisson equation for pressure
def pressure_poisson(p, b, dx, dy):
    pn = np.empty_like(p)
    for _ in range(50):  # Iterations for convergence
        pn[:] = p
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                           (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b)
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = L
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = L
    return p

# Update velocity fields
def update_velocity(u, v, p, rho, nu, dt, dx, dy):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    un[:], vn[:] = u, v
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 * 
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy**2 * 
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 * 
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy**2 * 
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
    apply_boundary_conditions(u, v)

# Time-stepping loop
for n in range(nt):
    b = build_b(u, v, rho, dt, dx, dy)
    p = pressure_poisson(p, b, dx, dy)
    update_velocity(u, v, p, rho, nu, dt, dx, dy)

# Visualization
plt.quiver(u[::2, ::2], v[::2, ::2])
plt.title("Velocity Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

