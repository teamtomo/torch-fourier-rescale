import numbers
from math import ceil, floor

import numpy as np
import torch
import torch.nn.functional as F



def fourier_rescale_3d(
    image: torch.Tensor,
    source_spacing: float | tuple[float, float, float] | None = None,
    target_spacing: float | tuple[float, float, float] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    preserve_mean: bool = True,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """Rescale 3D image(s) from `source_spacing` to `target_spacing` or to `target_shape`.

    Rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT).

    Parameters
    ----------
    image: torch.Tensor
        `(..., d, h, w)` array of image data
    source_spacing: float | tuple[float, float, float] | None
        Pixel spacing in the input image. Required if target_spacing is specified.
        If None when target_shape is specified, assumes uniform spacing of 1.0.
    target_spacing: float | tuple[float, float, float] | None
        Pixel spacing in the output image. Mutually exclusive with target_shape.
    target_shape: tuple[int, int, int] | None
        Target spatial dimensions (depth, height, width) for the output image.
        Mutually exclusive with target_spacing.
    preserve_mean: bool = True
        Ensure that the mean (DC component) of the array is preserved after rescaling.

    Returns
    -------
    rescaled_image, (new_spacing_d, new_spacing_h, new_spacing_w)
    """
    if target_spacing is None and target_shape is None:
        raise ValueError("Either target_spacing or target_shape must be specified")
    if target_spacing is not None and target_shape is not None:
        raise ValueError("Only one of target_spacing or target_shape can be specified")
    
    # Handle source_spacing
    if source_spacing is None:
        if target_spacing is not None:
            raise ValueError("source_spacing is required when target_spacing is specified")
        # Default to uniform spacing of 1.0 when using target_shape
        source_spacing = (1.0, 1.0, 1.0)
    elif isinstance(source_spacing, int | float | numbers.Real):
        source_spacing = (source_spacing, source_spacing, source_spacing)
    
    # If only target_shape is specified, we can skip spacing calculations
    if target_shape is not None and target_spacing is None:
        # No need to calculate target_spacing or check if spacings are close
        pass
    else:
        # If target_shape is specified along with source_spacing, calculate target_spacing
        if target_shape is not None:
            source_shape = image.shape[-3:]
            target_spacing = tuple(
                src_sp * (src_sh / tgt_sh)
                for src_sp, src_sh, tgt_sh in zip(source_spacing, source_shape, target_shape)
            )
        elif isinstance(target_spacing, int | float | numbers.Real):
            target_spacing = (target_spacing, target_spacing, target_spacing)
        
        if np.allclose(source_spacing, target_spacing):
            return image, source_spacing

    # place image center at array indices [0, 0, 0] and compute centered rfft3
    image = torch.fft.fftshift(image, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # Calculate target shape if using spacing
    if target_shape is None:
        # Calculate target shape from spacing
        source_shape = image.shape[-3:]
        target_shape = tuple(
            int(np.round(src_sh * (src_sp / tgt_sp)))
            for src_sh, src_sp, tgt_sp in zip(source_shape, source_spacing, target_spacing)
        )
    
    # Fourier pad/crop
    dft = fourier_rescale_rfft_3d(
        dft=dft,
        image_shape=image.shape[-3:],
        target_shape=target_shape,
    )
    new_shape = target_shape

    # transform back to real space and recenter
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))
    if preserve_mean:
        # we changed the number of elements in the FT so set norm='forward' to deactivate
        # default fft normalization by 1/n and normalise by the correct factor
        rescaled_image = torch.fft.irfftn(dft, dim=(-3, -2, -1), s=new_shape, norm="forward")
        rescaled_image = rescaled_image * (1 / np.prod(image.shape[-3:]))
    else:
        rescaled_image = torch.fft.irfftn(dft, dim=(-3, -2, -1), s=new_shape)
    rescaled_image = torch.fft.ifftshift(rescaled_image, dim=(-3, -2, -1))

    # Calculate new spacing after rescaling
    source_spacing = np.array(source_spacing, dtype=np.float32)
    new_spacing = tuple(
        src_sp * (src_sh / new_sh)
        for src_sp, src_sh, new_sh in zip(source_spacing, image.shape[-3:], new_shape)
    )

    return rescaled_image, tuple(new_spacing)


def fourier_rescale_rfft_3d(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Rescale a 3D rfft by padding or cropping to achieve target shape.
    
    Parameters
    ----------
    dft : torch.Tensor
        The rfftn result with fftshift applied to non-rfft dimensions
    image_shape : tuple[int, int, int]
        Original image shape (d, h, w)
    target_shape : tuple[int, int, int]
        Target image shape (d, h, w)
        
    Returns
    -------
    torch.Tensor
        The rescaled DFT
    """
    d, h, w = image_shape
    target_d, target_h, target_w = target_shape
    
    # Handle depth dimension
    if target_d > d:
        dft = _fourier_pad_d_shape(dft, source_depth=d, target_depth=target_d)
    elif target_d < d:
        dft = _fourier_crop_d_shape(dft, source_depth=d, target_depth=target_d)
    
    # Handle height dimension
    if target_h > h:
        dft = _fourier_pad_h_shape(dft, source_height=h, target_height=target_h)
    elif target_h < h:
        dft = _fourier_crop_h_shape(dft, source_height=h, target_height=target_h)
    
    # Handle width dimension  
    if target_w > w:
        dft = _fourier_pad_w_shape(dft, source_width=w, target_width=target_w)
    elif target_w < w:
        dft = _fourier_crop_w_shape(dft, source_width=w, target_width=target_w)
    
    return dft


def _fourier_crop_d(dft: torch.Tensor, image_depth: int, target_fftfreq: float):
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(image_depth))
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_d = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    new_d = torch.count_nonzero(idx_d)
    return dft[..., idx_d, :, :], new_nyquist, new_d


def _fourier_crop_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(image_height))
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_h = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    new_h = torch.count_nonzero(idx_h)
    return dft[..., :, idx_h, :], new_nyquist, new_h


def _fourier_crop_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    frequencies = torch.fft.rfftfreq(image_width)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_w = frequencies <= new_nyquist
    new_w = (
        torch.count_nonzero(idx_w) + torch.count_nonzero(frequencies < new_nyquist) - 1
    )
    return dft[..., :, :, idx_w], new_nyquist, new_w


def _fourier_pad_d(dft: torch.Tensor, image_depth: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_depth
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = (
        ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    )
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = (dft.shape[-3] // 2) + 1
    pad_d = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(
        dft,
        pad=(0, 0, 0, 0, pad_d, pad_d - (1 if image_depth % 2 == 1 else 0)),
        mode="constant",
        value=0,
    )
    new_d = dft.shape[-3]
    return dft, new_nyquist, new_d


def _fourier_pad_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_height
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = (
        ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    )
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = (dft.shape[-2] // 2) + 1
    pad_h = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(
        dft,
        pad=(0, 0, pad_h, pad_h - (1 if image_height % 2 == 1 else 0)),
        mode="constant",
        value=0,
    )
    new_h = dft.shape[-2]
    return dft, new_nyquist, new_h


def _fourier_pad_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_width
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = (
        ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    )
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = dft.shape[-1]
    pad_w = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(dft, pad=(0, pad_w), mode="constant", value=0)
    new_w = image_width + 2 * pad_w - (1 if image_width % 2 == 1 else 0)
    return dft, new_nyquist, new_w


def _fourier_crop_d_shape(dft: torch.Tensor, source_depth: int, target_depth: int):
    """Crop depth dimension to target depth."""
    current_d = dft.shape[-3]
    
    # Calculate crop amounts
    total_crop = current_d - target_depth
    crop_start = total_crop // 2
    crop_end = total_crop - crop_start
    
    # Crop the DFT
    if crop_end > 0:
        return dft[..., crop_start:-crop_end, :, :]
    else:
        return dft[..., crop_start:, :, :]


def _fourier_crop_h_shape(dft: torch.Tensor, source_height: int, target_height: int):
    """Crop height dimension to target height."""
    current_h = dft.shape[-2]
    
    # Calculate crop amounts
    total_crop = current_h - target_height
    crop_start = total_crop // 2
    crop_end = total_crop - crop_start
    
    # Crop the DFT
    if crop_end > 0:
        return dft[..., :, crop_start:-crop_end, :]
    else:
        return dft[..., :, crop_start:, :]


def _fourier_crop_w_shape(dft: torch.Tensor, source_width: int, target_width: int):
    """Crop width dimension to target width (rfft dimension)."""
    # For rfft, we need to handle the positive frequencies only
    # Calculate how many positive frequencies we need
    target_positive_freqs = (target_width // 2) + 1
    
    # Crop to keep only the required frequencies
    return dft[..., :, :, :target_positive_freqs]


def _fourier_pad_d_shape(dft: torch.Tensor, source_depth: int, target_depth: int):
    """Pad depth dimension to target depth."""
    current_d = dft.shape[-3]
    total_pad = target_depth - current_d
    
    # For even/odd handling
    pad_start = total_pad // 2
    pad_end = total_pad - pad_start
    
    # Adjust for odd source depth
    if source_depth % 2 == 1:
        pad_end = pad_end - 1
    
    return F.pad(dft, pad=(0, 0, 0, 0, pad_start, pad_end), mode="constant", value=0)


def _fourier_pad_h_shape(dft: torch.Tensor, source_height: int, target_height: int):
    """Pad height dimension to target height."""
    current_h = dft.shape[-2]
    total_pad = target_height - current_h
    
    # For even/odd handling
    pad_start = total_pad // 2
    pad_end = total_pad - pad_start
    
    # Adjust for odd source height
    if source_height % 2 == 1:
        pad_end = pad_end - 1
    
    return F.pad(dft, pad=(0, 0, pad_start, pad_end), mode="constant", value=0)


def _fourier_pad_w_shape(dft: torch.Tensor, source_width: int, target_width: int):
    """Pad width dimension to target width (rfft dimension)."""
    # For rfft, calculate how many positive frequencies we need
    current_positive_freqs = dft.shape[-1]
    target_positive_freqs = (target_width // 2) + 1
    pad_w = target_positive_freqs - current_positive_freqs
    
    return F.pad(dft, pad=(0, pad_w), mode="constant", value=0)
