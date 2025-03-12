import pytest
import torch.nn.functional as F

from torch_fourier_rescale import fourier_rescale_2d, fourier_rescale_3d


def test_fourier_upscale_2d(circle):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=0.5
    )
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (56, 56)

    # test upscale with uneven image
    rescaled, new_spacing = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (58, 58)


def test_fourier_downscale_2d(circle):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=2
    )
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (14, 14)

    # test downscale with uneven image
    rescaled, new_spacing = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (14, 14)


def test_fourier_upscale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=0.5
    )
    assert tuple(sphere.shape) == (14, 14, 14)
    assert tuple(rescaled.shape) == (28, 28, 28)

    # test upscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (30, 30, 30)


def test_fourier_downscale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=2
    )
    assert tuple(sphere.shape) == (14, 14, 14)
    assert tuple(rescaled.shape) == (6, 6, 6)

    # test downscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (8, 8, 8)


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_2d_mean(circle, spacing):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() == pytest.approx(circle.mean())

    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(circle.mean())


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_3d_mean(sphere, spacing):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() != pytest.approx(sphere.mean())

    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(sphere.mean())
