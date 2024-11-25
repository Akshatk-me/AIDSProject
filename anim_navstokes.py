import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation setup
nx, ny = 50, 50
dx, dy = 2 / (nx - 1), 2 / (ny - 1)
rho, dt = 1.0, 0.01
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
nit = 50

# Define the simulation functions: build_b, pressure_poisson, and the time step
def build_b(u, v, rho, dt, dx, dy):
    b = (rho * (1 / dt *
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, b, dx, dy, nit):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2 * (dx**2 + dy**2)))
        # Ensure boundary conditions are consistent
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = Lx
        p[0, :] = p[1, :]    # dp/dy = 0 at y = Ly
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 0
    return p


def velocity_step(u, v, p, rho, dt, dx, dy):
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                     (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                     (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * 
                     (p[1:-1, 2:] - p[1:-1, :-2]) +
                     dt / dx**2 * 
                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     dt / dy**2 * 
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                     (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                     (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * 
                     (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     dt / dx**2 * 
                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     dt / dy**2 * 
                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
    u[:, -1] = 0
    u[:, 0] = 0
    v[-1, :] = 0
    v[0, :] = 0
    return u, v

# Animation setup
fig, ax = plt.subplots(figsize=(7, 6))
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
quiver = ax.quiver(X, Y, u, v)
pressure_plot = ax.imshow(p, origin='lower', cmap='coolwarm', alpha=0.6)

def update(frame):
    global u, v, p
    b = build_b(u, v, rho, dt, dx, dy)
    p = pressure_poisson(p, b, dx, dy, nit)
    u, v = velocity_step(u, v, p, rho, dt, dx, dy)
    quiver.set_UVC(u, v)
    pressure_plot.set_array(p)
    return quiver, pressure_plot

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()

