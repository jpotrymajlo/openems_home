import numpy as np
import os

from openEMS.physical_constants import *
from CSXCAD import ContinuousStructure
from openEMS import openEMS

# =========================================================
# PARAMETRY
# =========================================================
f0 = 435e6
c0 = C0
wavelength = c0 / f0

monopole_h = 5/8 * wavelength
radius = 0.005

mesh_step = wavelength / 20

height_offset = 1.0
feed_gap = mesh_step   # 🔴 KLUCZ: szczelina zasilania
radial_len = wavelength / 4

# =========================================================
# STRUKTURA
# =========================================================
CSX = ContinuousStructure()
metal = CSX.AddMetal('PEC')

# =========================================================
# SIATKA
# =========================================================
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1.0)

z_feed_low  = height_offset
z_feed_high = height_offset + feed_gap
z_monopole_start = z_feed_high
z_monopole_end   = z_monopole_start + monopole_h

mesh.AddLine('x', [-wavelength, 0, wavelength])
mesh.AddLine('y', [-wavelength, 0, wavelength])

mesh.AddLine('z', [
    -wavelength,
    0,
    z_feed_low - mesh_step,  # 🔴 KLUCZ: linia siatki tuż poniżej dolnego końca szczeliny
    z_feed_low,
    z_feed_high,
    z_monopole_start,
    z_monopole_end,
    z_monopole_end + wavelength
])

mesh.SmoothMeshLines('x', mesh_step, 1.4)
mesh.SmoothMeshLines('y', mesh_step, 1.4)
mesh.SmoothMeshLines('z', mesh_step, 1.4)

# =========================================================
# GROUND (PEC)
# =========================================================
ground = CSX.AddMetal('GND')

ground.AddBox(
    priority=0,
    start=[-2*wavelength, -2*wavelength, 0],
    stop =[ 2*wavelength,  2*wavelength, 0]
)

# =========================================================
# MONOPOLE (CYLINDER)
# =========================================================
metal.AddCylinder(
    priority=10,
    start=[0, 0, z_monopole_start],
    stop =[0, 0, z_monopole_end],
    radius=radius
)
metal.AddCylinder(
    priority=10,
    start=[0,0,z_feed_low - feed_gap],  # 🔴 KLUCZ: dolny koniec szczeliny tuż poniżej z_feed_low
    stop =[0,0,z_feed_low],
    radius=radius
)
# =========================================================
# PRZECIWWAGI (NIE DOTYKAJĄ FEED POINT!)
# =========================================================
#radials = [
#    [ radial_len, 0, height_offset],
#    [-radial_len, 0, height_offset],
#    [0,  radial_len, height_offset],
#    [0, -radial_len, height_offset],
#]

#for r in radials:
#    metal.AddCylinder(
#        priority=10,
#        start=[0, 0, height_offset],
#        stop =[r[0], r[1], height_offset],
#        radius=radius
#    )

# =========================================================
# Dump pola
# =========================================================
dump_freq = CSX.AddDump('Efield_freq', dump_type=10, dump_mode=0, file_type=1)
dump_freq.SetFrequency([f0])
dump_freq.AddBox(
    [-wavelength, -wavelength, height_offset],
    [ wavelength,  wavelength, height_offset + monopole_h])



# =========================================================
# FDTD
# =========================================================
FDTD = openEMS(EndCriteria=1e-4)

# =========================================================
# BOUNDARY CONDITIONS - FREE SPACE
# =========================================================
FDTD.SetBoundaryCond(['MUR','MUR','MUR','MUR','MUR','MUR'])

# =========================================================
# FDTD + LUMPED PORT (POPRAWNY GAP FEED)
# =========================================================
FDTD.SetCSX(CSX)
FDTD.SetGaussExcite(f0, f0/2)

feed = FDTD.AddLumpedPort(
    1,        # ID
    50,
    [0, 0, z_feed_low],   # dolny koniec szczeliny
    [0, 0, z_feed_high],  # górny koniec szczeliny
    "z",
    1.0,
    L=0.5e-9,
    priority=5,
    edges2grid='z'
)

# =========================================================
# RUN + VISUALIZATION
# =========================================================
Sim_Path = "monopole_view"
os.makedirs(Sim_Path, exist_ok=True)

output_fn = os.path.join(Sim_Path, "geometry.xml")
CSX.Write2XML(output_fn)

#os.system(f"/home/jacek/opt/openEMS/bin/AppCSXCAD {output_fn}")

os.chdir(Sim_Path)
FDTD.Run('.')
os.chdir('..')

# =========================
# ANALIZA PORTU
# =========================
feed.CalcPort(Sim_Path, np.array([f0]))

z_in = feed.uf_tot/feed.if_tot
p_in = 0.5 * feed.uf_tot * np.conj(feed.if_tot)
S11 = np.sqrt(feed.P_ref / feed.P_inc)
SWR = (1 + np.abs(S11)) / (1 - np.abs(S11))

print(f"uf_ref = {feed.uf_ref}")
print(f"if_tot = {feed.if_tot}")
print(f"if_ref = {feed.if_ref}")
print(f"ZL_ref = {feed.Z_ref}")
print(f"P_inc = {feed.P_inc}")
print(f"P_ref = {feed.P_ref}")
print(f"P_acc = {feed.P_acc}")
print(f"uf_inc = {feed.uf_inc}")
print(f"uf_tot = {feed.uf_tot}")
print(f"S11 = {S11}")
print(f"SWR = {SWR}")
print(f"Z in @ (f0/1e6:.0f) MHz = {z_in[0]:.1f} Ohm")
print(f"P_in (symulacja) = {np.real(p_in[0]):.4f} W")