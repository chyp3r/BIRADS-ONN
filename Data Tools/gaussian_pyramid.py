from __future__ import annotations

import torch.nn.functional as F

from kornia.core import Tensor,tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.filters import filter2d,gaussian_blur2d


def gaussian_kernel() -> Tensor:
    """
    Create a gaussian kernel tensor

    Returns
        tensor: A gaussian kernel tensor
    """
    return (
        tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )
        / 256.0
    )

def pyr_down(input: Tensor, border_type: str = 'reflect', align_corners: bool = False, factor: float = 2.0, blur: bool = False) -> Tensor:
    """
    Create a downscale picture

    Args
        input: Image tensor
        border_type: Border type for image (Fore more information check kornia docs)
        align_corners: Align for image corners (Fore more information check kornia docs)
        factor: Downscale factor
        blur: Blur for image

    Returns
        out: New image
    """
        
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = gaussian_kernel()
    _, _, height, width = input.shape

    input: Tensor = filter2d(input, kernel, border_type)

    out: Tensor = F.interpolate(
        input,
        size=(int(float(height) / factor), int(float(width) // factor)),
        mode='bilinear',
        align_corners=align_corners,
    )
    
    if blur == True:
        out: Tensor = gaussian_blur2d(out, (3, 3),(3, 3), border_type)

    return out

def pyr_up(input: Tensor, border_type: str = 'reflect', align_corners: bool = False, blur: bool = False) -> Tensor:
    """
    Create a upscale picture

    Args
        input: Image tensor
        border_type: Border type for image (Fore more information check kornia docs)
        align_corners: Align for image corners (Fore more information check kornia docs)
        blur: Blur for image

    Returns
        out: New image
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = gaussian_kernel()
    _, _, height, width = input.shape

    input: Tensor = filter2d(input, kernel, border_type)

    out: Tensor = F.interpolate(
        input, 
        size=(height * 2, width * 2), 
        mode='bilinear', 
        align_corners=align_corners)

    return out

def pyr_merge_down(input: Tensor, border_type: str = 'reflect', align_corners: bool = False, factor: float = 2.0, blur: bool = False) -> Tensor:
    """
    Create a merged down picture

    Args
        input: Image tensor
        border_type: Border type for image (Fore more information check kornia docs)
        align_corners: Align for image corners (Fore more information check kornia docs)
        factor: Downscale factor
        blur: Blur for image

    Returns
        input: Merged image

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = gaussian_kernel()
    _, _, height, width = input.shape

    input: Tensor = filter2d(input, kernel, border_type)
        
    out: Tensor = F.interpolate(
        input,
        size=(int(float(height) / factor), int(float(width) // factor)),
        mode='bilinear',
        align_corners=align_corners,
    )

    if blur == True:
        out: Tensor = gaussian_blur2d(out, (3, 3),(3, 3), border_type)

    y = int(out.shape[2]/2)
    x = int(out.shape[3]/2)

    input[:,:,y:(y+out.shape[2]),x:(x+out.shape[3])] = out
    return input

def build_pyramid(input: Tensor, max_level: int, border_type: str = 'reflect', align_corners: bool = False, blur: bool = False) -> list[Tensor]:
    """
    Create a pyramid

    Args
        input: Image tensor
        max_level: Image count for pyramid
        border_type: Border type for image (Fore more information check kornia docs)
        align_corners: Align for image corners (Fore more information check kornia docs)
        blur: Blur for image

    Returns
        pyramid: A gaussian pyramid

    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    pyramid: list[Tensor] = []
    pyramid.append(input)

    # For downscale pyramid
    for _ in range(max_level - 1):
        img_curr: Tensor = pyramid[-1]
        img_down: Tensor = pyr_down(img_curr, border_type, align_corners,blur = blur)
        pyramid.append(img_down)
    
    # For upscale pyramid 
    # for _ in range(max_level - 1):
    #     img_curr: Tensor = pyramid[-1]
    #     img_down: Tensor = pyr_up(img_curr, border_type, align_corners,blur = blur)
    #     pyramid.append(img_down)

    return pyramid


