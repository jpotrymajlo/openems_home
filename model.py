import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyvista as pv

from openEMS import openEMS
from openEMS.physical_constants import *
from CSXCAD import ContinuousStructure


# =========================
# FUNKCJA SNAP
# =========================
def snap_to_mesh(value, mesh_array):
    return mesh_array[np.argmin(np.abs(mesh_array - value))]

# =========================
# PARAMETRY SYMULACJI
# =========================
f0 = 435e6
c0 = 299792458
lambda0 = c0 / f0
dx = lambda0 / 20   # dokładniejsza siatka

# =========================
# INICJALIZACJA
# =========================
CSX = ContinuousStructure()
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1)

x = np.arange(-3, 3, dx)
y = np.arange(0, 5, dx)
z = np.arange(0, 3, dx)

mesh.AddLine('x', x)
mesh.AddLine('y', y)
mesh.AddLine('z', z)

# =========================
# MATERIAŁY
# =========================
concrete = CSX.AddMaterial('concrete', epsilon=5.0, kappa=0.02)
glass = CSX.AddMaterial('glass', epsilon=6.0)
human = CSX.AddMaterial('human', epsilon=52, kappa=1.2)
metal = CSX.AddMetal('PEC')

# =========================
# GEOMETRIA
# =========================
wall_y = 2.5

concrete.AddBox(priority=1, start=[-2, 0, 0], stop=[2, 2.5, 0.2])
concrete.AddBox(priority=1, start=[-2, 0, 2.3], stop=[2, 2.5, 2.5])
concrete.AddBox(priority=2, start=[-2, wall_y, 0], stop=[2, wall_y+0.25, 2.5])

glass.AddBox(priority=5, start=[-0.5, wall_y, 1], stop=[0.5, wall_y+0.25, 2])

human.AddCylinder(priority=10,
                  start=[0, 1.0, 0],
                  stop=[0, 1.0, 1.75],
                  radius=0.15)

# =========================
# ANTENA (1 m nad podłogą)
# =========================
antenna_length = 0.625 * lambda0

antenna_x = snap_to_mesh(1.0, x)
antenna_y = snap_to_mesh(wall_y + 1.5, y)

antenna_z0 = snap_to_mesh(1.0, z)
antenna_z1 = snap_to_mesh(1.0 + antenna_length, z)

metal.AddCylinder(priority=20,
                  start=[antenna_x, antenna_y, antenna_z0],
                  stop=[antenna_x, antenna_y, antenna_z1],
                  radius=0.005)

# =========================
# PRZECIWWAGI (radiale)
# =========================
radial_len = 0.25
z_rad = antenna_z0

metal.AddCylinder(priority=15,
                  start=[antenna_x, antenna_y, z_rad],
                  stop=[snap_to_mesh(antenna_x + radial_len, x), antenna_y, z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x, antenna_y, z_rad],
                  stop=[snap_to_mesh(antenna_x - radial_len, x), antenna_y, z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x, antenna_y, z_rad],
                  stop=[antenna_x, snap_to_mesh(antenna_y + radial_len, y), z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x, antenna_y, z_rad],
                  stop=[antenna_x, snap_to_mesh(antenna_y - radial_len, y), z_rad],
                  radius=0.003)

# =========================
# PORT
# =========================
z0 = antenna_z0
z1 = snap_to_mesh(antenna_z0 + 2*dx, z)

feed = FDTD.AddLumpedPort(
    CSX,
    1,
    [antenna_x, antenna_y, z0],
    [antenna_x, antenna_y, z1],
    'z',
    50
)

# =========================
# POBUDZENIE
# =========================
FDTD.SetGaussExcite(f0, f0/2)

# =========================
# DUMP POLA
# =========================
dump = CSX.AddDump('Efield')
dump.AddBox(start=[-3, 0, 0], stop=[3, 5, 3])

# =========================
# GRANICE
# =========================
FDTD.SetBoundaryCond(['PML_8']*6)

# =========================
# SYMULACJA
# =========================
Sim_Path = 'sim_plot'
os.makedirs(Sim_Path, exist_ok=True)

os.chdir(Sim_Path)
FDTD.Run('.')
os.chdir('..')

# =========================
# ANALIZA PORTU
# =========================
port = feed.CalcPort(Sim_Path, f0)
print("Symulacja zakończona")

# =========================
# WCZYTANIE VTK
# =========================
file_vtr = Sim_Path + '/Efield_0000000000.vtr'
mesh_vtk = pv.read(file_vtr)

print("Dostępne pola:", mesh_vtk.array_names)

# =========================
# POLE E
# =========================
E = mesh_vtk['E-Field']   # wektor [Ex, Ey, Ez]
E_mag = np.linalg.norm(E, axis=1)

points = mesh_vtk.points
xv = points[:,0]
yv = points[:,1]
zv = points[:,2]

# =========================
# GĘSTOŚĆ MOCY
# =========================
eta0 = 377
S = E_mag**2 / (2 * eta0)

# =========================
# PRZEKRÓJ (1.5 m)
# =========================
mask = np.abs(zv - 1.5) < 0.05

x2 = xv[mask]
y2 = yv[mask]
E2 = E_mag[mask]
S2 = S[mask]

# =========================
# WYKRES E
# =========================
plt.figure(figsize=(8,6))
plt.tricontourf(x2, y2, E2, levels=50)
plt.colorbar(label='E [V/m]')
plt.title('Pole E (1.5 m)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.show()

# =========================
# WYKRES S
# =========================
plt.figure(figsize=(8,6))
plt.tricontourf(x2, y2, S2, levels=50)
plt.colorbar(label='S [W/m²]')
plt.title('Gęstość mocy')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.show()

# =========================
# PRZEKROCZENIA
# =========================
limit = 2.0
danger = S2 > limit

plt.figure(figsize=(8,6))
plt.scatter(x2[danger], y2[danger], c='red', s=5)
plt.title('Przekroczenia normy ICNIRP')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.show()
