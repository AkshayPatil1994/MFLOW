import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#
# User Input Data
#
rans_data_location = '/mnt/data/alpatil/refmap/campus/TUDcampus/lod2p2/binary_data/data'
z_height = 10                       # Height at which the slice is taken
# SKIP datasets
skip_RANS = False
# Data storage parameters
save_interp_data = True            # Save the interpolated data to file
# Reference velocities at z = 10m for normalization
Uref_rans = 7.69
Uref_les =  7.609
# Rotation origins for LES data (identified by matching building footprints in the OBJ model...)
rotation_origin_x = 1204.015
rotation_origin_y = 1272.635
# Parallelization parameters
procx, procy = 4, 2
# Unified grid parameters
x_lo, x_hi = 1250,1500              # X min-max limits 
y_lo, y_hi = 1100,1400              # Y min-max limits
# The settings below come from LES simulation setup, can be changed based on the LES setup
dx, dy =  min(np.round(3400/896,2), np.round(2500/512)), min(np.round(3400/896,2), np.round(2500/512))
x_common = np.arange(start=x_lo,stop=x_hi+dx,step=dx)
y_common = np.arange(start=y_lo,stop=y_hi+dy,step=dy)
X, Y = np.meshgrid(x_common,y_common,indexing='ij')
if(save_interp_data):
    np.savetxt('truncated_data/x_common_grid.txt',x_common,header='x')
    np.savetxt('truncated_data/y_common_grid.txt',y_common,header='y')
# Debugging flags
verbose = False
# HARD CODED TRANSLATION FOR RANS
x_add, y_add = 1328, 1213
# HARD CODED LES WIND DIRECTIONS AND EXP NUMBERS
les_collocation_points = [0.59,4.37,16.57,37.28,51.62,57.55,61.7,65.28,69.16,73.95,80.24,88.15,97.2,105.51,110.89,114.42,117.54,121.19,126.45,133.31,139.9,149.34,163.13,172.08,177.08,180.76,185.65,192.73,203.04,216.59,237.0,261.88,275.8,283.3,290.15,298.0,310.01,325.46,336.53,341.85,345.11,347.46,349.42,351.3,353.16,355.14,357.14,358.86]
exp_numbers = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
#
# MAIN 
#
print("=="*25)
print("Interpolating RANS & LES data")
print(f"Unified grid has a resolution of {dx} m in x and y")
print(f"    ASSUMPTIONS:")
print(f"1. RANS data at 1-degree resolution")
print(f"2. LES data at sparse interval within the RANS range specified")
print("=="*25)
#
# First load RANS data on the common grid
#
# First load the static grid
filename = f'{rans_data_location}/x_{z_height}.bin'
with open(filename,'rb') as f:
    f.seek(0)
    x_rans = np.fromfile(f,dtype=np.float64)
    f.close()
filename = f'{rans_data_location}/y_{z_height}.bin'
with open(filename,'rb') as f:
    f.seek(0)
    y_rans = np.fromfile(f,dtype=np.float64)
    f.close()
# Apply translation to RANS coordinates
x_rans = x_rans + x_add
y_rans = y_rans + y_add
# Load all the angles and interpolate the data onto the common grid
if(not skip_RANS):
    for angle in tqdm(range(1,361),desc="Interpolating RANS data"):
        rans_filename = f'{rans_data_location}/Umag_{z_height}_{angle}.bin' 
        # Load the rans data file
        with open(rans_filename,'rb') as f:
            f.seek(0)
            Udata = np.fromfile(f,dtype=np.float64)/Uref_rans
            f.close()
        # Interpolate RANS data onto the equidistant grid
        U_rans_normalised = griddata((x_rans, y_rans), Udata, (X, Y), method='nearest')
        if(verbose and angle == 1):
            print(f"Verbose check - Size of U_rans_normalised: {U_rans_normalised.size}")
        if(save_interp_data):
            rans_outfilename = f'truncated_data/U_rans_normalised_{angle:03d}.txt'
            np.savetxt(rans_outfilename,U_rans_normalised,header='Umag')

# Load all LES data based on the available wind directions
for my_case in zip(les_collocation_points, tqdm(exp_numbers,desc="Interpolating LES data")):
    angle = my_case[0]
    case_index = my_case[1]

    # Assertion to check if file exists
    assert_file = f'../{case_index:03d}/analysis/data/zplane_{z_height}m_000.000.{case_index:03d}.nc'
    if(not os.path.isfile(assert_file)):
        print(f"ERROR: File {assert_file} not found. Skipping case {case_index}.")
    else:
        # First pass: determine global array size
        filename_first = f'../{case_index:03d}/analysis/data/zplane_{z_height}m_000.000.{case_index:03d}.nc'
        ds_first = xr.open_dataset(filename_first)
        ny_tile, nx_tile = ds_first['u'].shape
        ds_first.close()
        
        # Create global arrays for staggered velocity components
        u_global = np.full((procy * ny_tile, procx * nx_tile), np.nan)
        v_global = np.full((procy * ny_tile, procx * nx_tile), np.nan)
        w_global = np.full((procy * ny_tile, procx * nx_tile), np.nan)
        x_global = np.full(procx * nx_tile, np.nan)
        y_global = np.full(procy * ny_tile, np.nan)

        # Fill in each tile with the staggered velocity data
        for ix in range(0, procx):
            for iy in range(0, procy):
                filename = f'../{case_index:03d}/analysis/data/zplane_10m_{ix:03d}.{iy:03d}.{case_index:03d}.nc'        
                ds = xr.open_dataset(filename)
                xm = ds['xm'].values
                ym = ds['ym'].values
                u = ds['u'].values
                v = ds['v'].values
                w = ds['w'].values
                
                # Place tiles in global arrays (staggered)
                u_global[iy*ny_tile:(iy+1)*ny_tile, ix*nx_tile:(ix+1)*nx_tile] = u
                v_global[iy*ny_tile:(iy+1)*ny_tile, ix*nx_tile:(ix+1)*nx_tile] = v
                w_global[iy*ny_tile:(iy+1)*ny_tile, ix*nx_tile:(ix+1)*nx_tile] = w
                x_global[ix*nx_tile:(ix+1)*nx_tile] = xm
                y_global[iy*ny_tile:(iy+1)*ny_tile] = ym
                
                ds.close()

        # Now interpolate the entire global field to cell centers
        u_center = 0.5 * (u_global[:, :-1] + u_global[:, 1:])  
        v_center = 0.5 * (v_global[:-1, :] + v_global[1:, :])  
        
        # Determine final common dimensions
        ny_final = min(u_center.shape[0], v_center.shape[0])
        nx_final = min(u_center.shape[1], v_center.shape[1])
        
        # Trim all arrays to common dimensions
        u_center = u_center[:ny_final, :nx_final]
        v_center = v_center[:ny_final, :nx_final]
        w_center = w_global[:ny_final, :nx_final]
        
        # Calculate cell-center coordinates to match the trimmed arrays
        x_center = 0.5 * (x_global[:-1] + x_global[1:])[:nx_final]
        y_center = 0.5 * (y_global[:-1] + y_global[1:])[:ny_final]
        
        # Calculate velocity magnitude (normalised)
        Umag = np.sqrt(u_center**2 + v_center**2 + w_center**2) / Uref_les
        
        # Create 2D coordinate grids for rotation
        Xles, Yles = np.meshgrid(x_center, y_center)
        
        # Rotate LES grid to match RANS orientation
        theta = np.deg2rad(angle)
        Xles_rotated = (Xles - rotation_origin_x) * np.cos(theta) - (Yles - rotation_origin_y) * np.sin(theta) + rotation_origin_x
        Yles_rotated = (Xles - rotation_origin_x) * np.sin(theta) + (Yles - rotation_origin_y) * np.cos(theta) + rotation_origin_y

        # Interpolate the LES data onto the common grid (Ravel to match size)
        U_les_normalised = griddata((Xles_rotated.ravel(), Yles_rotated.ravel()), Umag.ravel(), (X, Y), method='nearest')

        # Save when prompted by the user
        if(save_interp_data):
            les_outfilename = f'truncated_data/U_les_normalised_{case_index:03d}.txt'
            np.savetxt(les_outfilename,U_les_normalised,header='Umag')