#!/usr/bin/env python3

import numpy as np
import torch

def one_hot_numpy(indices, depth, on_value=1, off_value=0, axis=-1, dtype=None):
    # Convert indices to numpy array for consistent handling
    indices = np.asarray(indices)
    
    # Determine the output shape
    out_shape = list(indices.shape) + [depth] if axis == -1 else [depth] + list(indices.shape)
    
    # Create a tensor filled with off_value
    result = np.full(out_shape, off_value, dtype=dtype if dtype else type(on_value))
    
    # Replace the off_value with on_value at the appropriate positions
    if axis == -1:
        # Last axis (default)
        for idx, index in np.ndenumerate(indices):
            if 0 <= index < depth:  # Ignore out-of-bound indices
                result[idx + (index,)] = on_value
    elif axis == 0:
        # First axis
        for idx, index in np.ndenumerate(indices):
            if -1 * depth <= index < depth:
                result[(index,) + idx] = on_value
    else:
        raise ValueError("Only axis=-1 or axis=0 are supported in this simplified implementation.")
    return result

def one_hot_torch(indices, depth, on_value=1, off_value=0, axis=-1, dtype=None):
    # Convert indices to a tensor
    indices = torch.tensor(indices, dtype=torch.long)
    
    # Determine the dtype of the output tensor
    dtype = dtype if dtype else torch.float32 if isinstance(on_value, float) else torch.int64

    # Create a mask for valid indices (-depth <= index < depth)
    valid_mask = (-1 * depth <= indices) & (indices < depth)
    # Adjust negative indices to positive range
    adjusted_indices = torch.where(indices < 0, indices + depth, indices)

    # Clamp invalid indices to 0 to avoid scatter errors
    safe_indices = torch.where(valid_mask, adjusted_indices, torch.zeros_like(indices))

    # Create the shape for the output tensor
    shape = list(indices.shape)
    if axis == -1:
        shape.append(depth)
    elif axis == 0:
        shape = [depth] + shape
    else:
        raise ValueError("Only axis=-1 or axis=0 are supported.")

    # Initialize the result tensor with off_value
    result = torch.full(shape, off_value, dtype=dtype)

    # Expand indices for scatter
    if axis == -1:
        indices_unsqueezed = safe_indices.unsqueeze(-1)
        result.scatter_(-1, indices_unsqueezed, on_value)
    elif axis == 0:
        indices_unsqueezed = safe_indices.unsqueeze(0)
        result.scatter_(0, indices_unsqueezed, on_value)
    
    # Ensure invalid indices remain off_value
    # if not valid_mask.all():
    #     result[~valid_mask] = off_value
    return result


# Examples

indices = [0, 1, 2]
depth = 3
print("#" * 30)
print(f"inputs: {indices=}, {depth=}")
print("numpy\n", one_hot_numpy(indices, depth))
print("torch\n", one_hot_torch(indices, depth))

indices = [0, 2, -1, 1]
depth = 3
print("#" * 30)
print(f"inputs: {indices=}, {depth=}")
print("numpy\n", one_hot_numpy(indices, depth, on_value=5.0, off_value=0.0))
print("torch\n", one_hot_torch(indices, depth, on_value=5.0, off_value=0.0))


indices = [[0, 2], [1, -1]]
depth = 3
print("#" * 30)
print(f"inputs: {indices=}, {depth=}")
print("numpy\n", one_hot_numpy(indices, depth))
print("torch\n", one_hot_torch(indices, depth))
