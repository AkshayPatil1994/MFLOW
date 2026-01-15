import numpy as np
import xarray as xr
from tqdm import tqdm
#
# INPUT DATA
#
save_output = False                                 # Whether to save output files
procx, procy = 4, 2                                 # Number of processors in x and y
exp_start = 27                                      # Experiment start number
exp_end = 50                                        # Experiment end number
slice_location = 10                                 # Vertical location in meters

#
# Helper function for loading solid points for masking
#
def load_solid_indices(filename):
    """
        Load solid cell indices from file. Returns array of (i, j, k) with 0-based indexing.
    """
    indices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 3:
                i, j, k = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1  # Convert to 0-based
                indices.append((i, j, k))
    return np.array(indices) if indices else np.empty((0, 3), dtype=int)
#
# Extract the masking points for the requested slice
#
def get_solid_mask_for_slice(solid_indices, k_lower, k_upper, nx_local, ny_local, ix, iy):
    """
        Get local (i, j) indices that are solid at the interpolation levels.
    """
    # Filter indices where k is at either interpolation level
    mask_k = (solid_indices[:, 2] == k_lower) | (solid_indices[:, 2] == k_upper)
    filtered = solid_indices[mask_k]

    # Map global indices to local processor indices
    i_start = ix * nx_local
    i_end = (ix + 1) * nx_local
    j_start = iy * ny_local
    j_end = (iy + 1) * ny_local

    # Filter to indices within this processor's domain
    mask_proc = ((filtered[:, 0] >= i_start) & (filtered[:, 0] < i_end) &
                 (filtered[:, 1] >= j_start) & (filtered[:, 1] < j_end))
    local_indices = filtered[mask_proc]

    # Convert to local coordinates
    local_i = local_indices[:, 0] - i_start
    local_j = local_indices[:, 1] - j_start

    return local_i, local_j

#
# MAIN ANALYSIS #
#
# First calculate the weight for interpolation 
# -- assumes z coordinate is identical for all experiments -- 
exp_num = exp_start
ix, iy = 0, 0
base_location = f'../{exp_num:03d}/analysis/data/'
filename = f'{base_location}tavg.{ix:03d}.{iy:03d}.{exp_num:03d}.nc'
print(f"## Calculating interpolation weight for vertical slice ##")         
ds = xr.open_dataset(filename)                        
zm_values = ds['zm'].values
# Sanity check for slice_location
if slice_location < zm_values[0] or slice_location > zm_values[-1]:
    raise ValueError(f"slice_location {slice_location} m is out of bounds ({zm_values[0]} m to {zm_values[-1]} m)")

zmslice_index = np.where(zm_values <= slice_location)[0][-1]
# Interpolate tavg to the exact location
z1m = zm_values[zmslice_index]
z2m = zm_values[zmslice_index + 1]
weight_m = (slice_location - z1m) / (z2m - z1m)
print(f"Center Interpoation Weight: {slice_location} m is {weight_m:.4f} between zm-index {zmslice_index} ({z1m} m) and {zmslice_index + 1} ({z2m} m)")
# Face values
zt_values = ds['zt'].values
ztslice_index = np.where(zt_values <= slice_location)[0][-1]
# Interpolate tavg to the exact location
z1t = zt_values[ztslice_index]
z2t = zt_values[ztslice_index + 1]
weight_t = (slice_location - z1t) / (z2t - z1t)
print(f"Face Interpoation Weight: {slice_location} m is {weight_t:.4f} between zt-index {ztslice_index} ({z1t} m) and {ztslice_index + 1} ({z2t} m)")

# LOOP OVER EXPERIMENTS
print(f"### Starting analysis for experiments {exp_start} to {exp_end} ###")
for exp_num in tqdm(range(exp_start, exp_end + 1), desc="Extracting Z-slices"):
    base_location = f'../{exp_num:03d}/analysis/data/'
    exp_root = f'../{exp_num:03d}/'

    # Load solid indices for this experiment
    solid_u = load_solid_indices(f'{exp_root}solid_u.txt')
    solid_v = load_solid_indices(f'{exp_root}solid_v.txt')
    solid_w = load_solid_indices(f'{exp_root}solid_w.txt')

    for ix in range(0, procx):
        for iy in range(0, procy):
            filename = f'{base_location}tavg.{ix:03d}.{iy:03d}.{exp_num:03d}.nc'
            output_filename = f'{base_location}zplane_{slice_location}m_{ix:03d}.{iy:03d}.{exp_num:03d}.nc'

            ds = xr.open_dataset(filename)

            # Get local grid dimensions
            nx_local = len(ds['xm'])
            ny_local = len(ds['ym'])

            ds_slice = xr.Dataset()
            # Interpolate U - cell center for z direction (u uses xm for i, yt for j)
            var_data = ds['u']
            slice_data = (1 - weight_m) * var_data.isel(zt=zmslice_index, drop=True) + weight_m * var_data.isel(zt=zmslice_index + 1, drop=True)
            # Mask solid cells for u (uses xm, yt grid)
            local_i, local_j = get_solid_mask_for_slice(solid_u, zmslice_index, zmslice_index + 1, nx_local, ny_local, ix, iy)
            slice_values = slice_data.values.copy()
            for li, lj in zip(local_i, local_j):
                if li < slice_values.shape[1] and lj < slice_values.shape[0]:  # u is (yt, xm)
                    slice_values[lj, li] = np.nan
            ds_slice['u'] = xr.DataArray(slice_values, dims=slice_data.dims, coords=slice_data.coords)

            # Interpolate V - cell center for z direction (v uses xt for i, ym for j)
            var_data = ds['v']
            slice_data = (1 - weight_m) * var_data.isel(zt=zmslice_index, drop=True) + weight_m * var_data.isel(zt=zmslice_index + 1, drop=True)
            # Mask solid cells for v (uses xt, ym grid)
            local_i, local_j = get_solid_mask_for_slice(solid_v, zmslice_index, zmslice_index + 1, len(ds['xt']), len(ds['ym']), ix, iy)
            slice_values = slice_data.values.copy()
            for li, lj in zip(local_i, local_j):
                if li < slice_values.shape[1] and lj < slice_values.shape[0]:  # v is (ym, xt)
                    slice_values[lj, li] = np.nan
            ds_slice['v'] = xr.DataArray(slice_values, dims=slice_data.dims, coords=slice_data.coords)

            # Interpolate W - cell face for z direction (w uses xt for i, yt for j)
            var_data = ds['w']
            slice_data = (1 - weight_t) * var_data.isel(zm=ztslice_index, drop=True) + weight_t * var_data.isel(zm=ztslice_index + 1, drop=True)
            # Mask solid cells for w (uses xt, yt grid)
            local_i, local_j = get_solid_mask_for_slice(solid_w, ztslice_index, ztslice_index + 1, len(ds['xt']), ny_local, ix, iy)
            slice_values = slice_data.values.copy()
            for li, lj in zip(local_i, local_j):
                if li < slice_values.shape[1] and lj < slice_values.shape[0]:  # w is (yt, xt)
                    slice_values[lj, li] = np.nan
            ds_slice['w'] = xr.DataArray(slice_values, dims=slice_data.dims, coords=slice_data.coords)

            # Write the dataset to file
            if save_output:
                ds_slice.to_netcdf(output_filename)

            ds.close()
    
  
