import torch
import numpy as np

# original code from https://github.com/poetrywanderer/CF-NeRF/blob/66918a9748c137e1c0242c12be7aa6efa39ece06/run_nerf_helpers.py#L382

def ause(unc_vec, err_vec, err_type='rmse'):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
       

    ###########################################

    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause