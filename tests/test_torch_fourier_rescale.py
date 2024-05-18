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