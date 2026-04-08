import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyvista as pv
import glob

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
dx_fine = lambda0 / 80

# =========================
# INICJALIZACJA
# =========================
CSX = ContinuousStructure()
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1)



# =========================
# MATERIAŁY
# =========================
concrete = CSX.AddMaterial('concrete', epsilon=5.0, kappa=0.02)
glass = CSX.AddMaterial('glass', epsilon=6.0, kappa=0.001)
metal = CSX.AddMetal('PEC')

# =========================
# GEOMETRIA
# =========================
wall_y = 2.5
wall_thickness = 0.25
glass_thickness = 0.006
pml_margin = 12*dx
x_min = -2 -pml_margin
x_max = 2 + pml_margin
y_min = 0 - pml_margin
y_max = wall_y + wall_thickness + 2.5 + pml_margin
z_min = 0 - pml_margin
z_max = 3.0 + pml_margin

x = np.arange(x_min, x_max, dx)
y = np.arange(y_min, y_max, dx)
z = np.arange(z_min, z_max, dx)

mesh.AddLine('x', x)
mesh.AddLine('y', y)
mesh.AddLine('z', z)

wall_y_inner = wall_y
wall_y_outer = wall_y + wall_thickness

glass_y_center = (wall_y_inner + wall_y_outer)/2.0
glass_y_start = glass_y_center - glass_thickness / 2.0
glass_y_end = glass_y_center + glass_thickness / 2.0

mesh.AddLine('y', np.arange(wall_y_inner, wall_y_outer, dx_fine))
mesh.AddLine('y', [glass_y_start, glass_y_end])
mesh.SmoothMeshLines('y', dx, 1.3)

#=======================
# Sciany
# ======================

concrete.AddBox(priority=1, start=[-2, 0, 0], stop=[2, dx, 2.5])
concrete.AddBox(priority=1, start=[-2, 0, 0], stop=[-2+dx, wall_y, 2.5])
concrete.AddBox(priority=1, start=[2-dx, 0, 0], stop=[2, wall_y, 2.5])
concrete.AddBox(priority=1, start=[-2, 0, 0], stop=[2, wall_y, 0.2])
concrete.AddBox(priority=1, start=[-2, 0, 2.3], stop=[2, wall_y, 2.5])
concrete.AddBox(priority=1, start=[-2, wall_y_inner, 0], stop=[2, wall_y, 2.5])
#===================================================
# Okno
#===================================================
glass.AddBox(priority=5, start=[-0.5, glass_y_start, 1.0], stop=[0.5, glass_y_end, 2.0])


# =========================
# ANTENA (1 m nad podłogą)
# =========================
antenna_length = 0.625 * lambda0

antenna_x = 0.0
antenna_y = wall_y_outer + 1.5
antenna_length = lambda0 * 5 / 8

mesh.AddLine('x', np.arange(antenna_x - 0.1, antenna_x+0.1, dx_fine))
mesh.AddLine('y', np.arange(antenna_y - 0.1, antenna_y+0.1, dx_fine))
mesh.SmoothMeshLines('x', dx, 1.3)
mesh.SmoothMeshLines('y', dx, 1.3)

#antenna_z0 = 1.0
#antenna_z1 = antenna_z0 + antenna_length
#mesh.AddLine('z', [antenna_z0, antenna_z1])
#mesh.SmoothMeshLines('z', dx, 1.3)

antenna_x_snapped = snap_to_mesh(antenna_x, mesh.GetLines('x'))
antenna_y_snapped = snap_to_mesh(antenna_y, mesh.GetLines('y'))

#antenna_z0_snapped = snap_to_mesh(antenna_z0, mesh.GetLines('z'))
#antenna_z1_snapped = snap_to_mesh(antenna_z1, mesh.GetLines('z'))


#metal.AddCylinder(priority=20,
#                  start=[antenna_x_snapped, antenna_y_snapped, antenna_z0_snapped],
#                  stop=[antenna_x_snapped, antenna_y_snapped, antenna_z1_snapped],
#                  radius=0.005)
                  
mesh.AddLine('z', [1.0])
mesh.SmoothMeshLines('z', dx, 1.3)

z_lines = mesh.GetLines('z')
antenna_z0_snapped = snap_to_mesh(1.0, z_lines)

idx = np.where(z_lines == antenna_z0_snapped)[0][0]
antenna_z1_snapped = z_lines[idx + int(antenna_length/dx)]
z_feed_0 = antenna_z0_snapped
z_feed_1 = z_lines[idx + 1]
# dolna część (do portu)
metal.AddCylinder(
    priority=20,
    start=[antenna_x_snapped, antenna_y_snapped, antenna_z0_snapped],
    stop=[antenna_x_snapped, antenna_y_snapped, z_feed_0],
    radius=0.005
)

# górna część (od portu w górę)
metal.AddCylinder(
    priority=20,
    start=[antenna_x_snapped, antenna_y_snapped, z_feed_1],
    stop=[antenna_x_snapped, antenna_y_snapped, antenna_z1_snapped],
    radius=0.005
)
# =========================
# PRZECIWWAGI (radiale)
# =========================
radial_len = 0.25 * lambda0
z_rad = antenna_z0_snapped
x_lines = mesh.GetLines('x')
y_lines = mesh.GetLines('y')

metal.AddCylinder(priority=15,
                  start=[antenna_x_snapped, antenna_y_snapped, z_rad],
                  stop=[snap_to_mesh(antenna_x_snapped + radial_len, x_lines), antenna_y_snapped, z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x_snapped, antenna_y_snapped, z_rad],
                  stop=[snap_to_mesh(antenna_x_snapped - radial_len, x_lines), antenna_y_snapped, z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x_snapped, antenna_y_snapped, z_rad],
                  stop=[antenna_x_snapped, snap_to_mesh(antenna_y_snapped + radial_len, y_lines), z_rad],
                  radius=0.003)

metal.AddCylinder(priority=15,
                  start=[antenna_x_snapped, antenna_y_snapped, z_rad],
                  stop=[antenna_x_snapped, snap_to_mesh(antenna_y_snapped- radial_len, y_lines), z_rad],
                  radius=0.003)

# =========================
# PORT
# =========================
#z_feed_0 = antenna_z0_snapped

# znajdź NAJBLIŻSZY następny punkt siatki
#z_lines = mesh.GetLines('z')
#idx = np.where(z_lines == z_feed_0)[0][0]

#z_feed_1 = z_lines[idx + 1]   # dokładnie 1 komórka!

feed = FDTD.AddLumpedPort(
    CSX,
    1,                # numer portu
    [antenna_x_snapped, antenna_y_snapped, z_feed_0],
    [antenna_x_snapped, antenna_y_snapped, z_feed_1],
    'z',50
)

# =========================
# POBUDZENIE
# =========================
FDTD.SetGaussExcite(f0, f0/2)

# =========================
# DUMP POLA
# =========================
dump_freq = CSX.AddDump('Efield_freq', dump_type=10, dump_mode=0, file_type=1)
dump_freq.SetFrequency([f0])
dump_freq.AddBox(start=[x_min, y_min, z_min], stop=[x_max, y_max, z_max])

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
port = feed.CalcPort(Sim_Path, np.array([f0]))

z_in = feed.uf_tot/feed.if_tot
p_in = 0.5 * feed.uf_tot * np.conj(feed.if_tot)
print(f"Z in @ (f0/1e6:.0f) MHz = {z_in[0]:.1f} Ohm")
print(f"P_in (symulacja) = {np.real(p_in[0]):.4f} W")

print("Symulacja zakończona")

# =========================
# WCZYTANIE h5
# =========================

h5_files = sorted(glob.glob('sim_plot/Efield_freq*.h5'))
h5_file = h5_files[0] 

with h5py.File(h5_file, 'r') as f:
    print("Klucze:", list(f.keys()))
    E_real = np.array(f['FieldData']['FD']['f0_real'])
    E_imag = np.array(f['FieldData']['FD']['f0_imag'])
    xf = np.array(f['Mesh']['x'])
    yf = np.array(f['Mesh']['y'])
    zf = np.array(f['Mesh']['z'])

E_complex = E_real +1j*E_imag

e_mag_complex = np.sqrt(np.abs(E_complex[:,:,:,0])**2 + np.abs(E_complex[:,:,:,1])**2 + np.abs(E_complex[:,:,:,2])**2)

p_tx = 25.0
p_sim = np.real(p_in[0])
if p_sim > 0:
    scale = np.sqrt(p_tx / p_sim)
else:
    scale = 1.0
    print("Moc < 0 skalowanie domyslne")
    
e_mag_scaled = e_mag_complex * scale



# wybór przekroju w wysokości z_target
z_target = 1.5
z_idx = np.argmin(np.abs(zf - z_target))

# przygotowanie punktów x,y i wartości e
# dopasowanie siatki do danych pola
xf_c = xf[:-1]
yf_c = yf[:-1]

X, Y = np.meshgrid(xf_c, yf_c, indexing='ij')
E_slice = e_mag_scaled[:, :, z_idx]

# flatten
x_flat = X.flatten()
y_flat = Y.flatten()
e_flat = E_slice.flatten()

# dodatkowe zabezpieczenie
assert len(x_flat) == len(e_flat)

plt.figure(figsize=(10,7))
levels_e = np.linspace(0, np.percentile(e_flat, 99), 60)
cs = plt.tricontourf(x_flat, y_flat, e_flat, levels=levels_e, cmap='jet')
plt.colorbar(cs, label='|E| V/m')

# rysowanie okna, ściany i anteny
plt.plot([-2,2,2,-2,-2], [0,0,wall_y,wall_y,0], 'white', linewidth=4, label='Pokoj')
plt.plot([-2,2], [wall_y, wall_y], 'cyan', linewidth=3, label='Sciana')
plt.plot([-2,2], [wall_y_outer, wall_y_outer], 'cyan', linewidth=3)
plt.plot([-0.5,0.5], [wall_y + 0.125, wall_y + 0.125], 'lime', linewidth=4, label='Okno')
plt.scatter(antenna_x_snapped, antenna_y_snapped, c='blue', s=60, zorder=10, label='Antena')
plt.tight_layout()
plt.show()

eta0 = 377.0
s_slice = e_slice**2 / (2*eta0)



