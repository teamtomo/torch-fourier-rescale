import numpy as np
import pytest
import torch
from torch_fourier_rescale import fourier_rescale_2d, fourier_rescale_3d


def test_fourier_rescale_2d(circle):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=0.5
    )
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (56, 56)


def test_fourier_rescale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=0.5
    )
    assert tuple(sphere.shape) == (14, 14, 14)
    assert tuple(rescaled.shape) == (28, 28, 28)


@pytest.mark.parametrize("dtype", [
    int,
    float,
    np.float32,
    np.float64,
    torch.float32,
    torch.float64
])
def test_pixel_spacing_scalar_dtypes(dtype, circle, sphere):
    # Smoke test - just verify functions don't crash with different scalar dtypes
    if dtype == int:
        source_spacing = 1
        target_spacing = 2
    elif dtype == float:
        source_spacing = 1.0
        target_spacing = 0.5
    elif dtype in [np.float32, np.float64]:
        source_spacing = dtype(1.0)
        target_spacing = dtype(0.5)
    else:  # torch dtypes
        source_spacing = torch.tensor(1.0, dtype=dtype).item()
        target_spacing = torch.tensor(0.5, dtype=dtype).item()
    
    # Test 2D with dtype for both source and target spacing
    rescaled_2d, new_spacing_2d = fourier_rescale_2d(
        image=circle, 
        source_spacing=source_spacing, 
        target_spacing=target_spacing
    )
    assert rescaled_2d is not None
    assert new_spacing_2d is not None
    
    # Test 3D with dtype for both source and target spacing
    rescaled_3d, new_spacing_3d = fourier_rescale_3d(
        image=sphere,
        source_spacing=source_spacing,
        target_spacing=target_spacing
    )
    assert rescaled_3d is not None
    assert new_spacing_3d is not None