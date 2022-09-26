"""
SRDataset for generating (downscaled, original) image pairs.
"""
import random
import numpy as np
from PIL import Image, ImageOps



class SRDataset():
    """
    SRDataset generates (downscaled, original) image pairs for training and
    testing Super-Resolution DNN models.

    Args:
        paths ([string]): paths to a collection of images.
        scale (int): downscaling ratio.
        tile_size (int): side-length of randomly-cropped squared tiles
            (for training only).
        training (bool): created for training or not.

    Returns:
        randomly accessible dataset
    """

    def __init__(self, paths, scale=2, tile_size=144,
                 training=True):
        super(SRDataset, self).__init__()
        self._paths = paths
        self._scale = scale
        self._tile_size = tile_size
        self._downscaled_size = tile_size // scale
        self._cache = {}
        self._cache_ds = {}
        self._training = training

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index, vis=False):
        # Get target image
        if index not in self._cache:
            orig = Image.open(self._paths[index])

            w, h = orig.size
            # Make sure the target image is divisible by scale on testing
            if not self._training:
                w_r = w % self._scale
                h_r = h % self._scale
                if w_r + h_r > 0:
                    orig = orig.crop((0, 0, w - w_r, h - h_r))
                    w -= w_r
                    h -= h_r

            # Downscale the image
            downscaled = orig.resize(
                (w // self._scale, h // self._scale), Image.ANTIALIAS)

            # Cache the results
            self._cache[index] = orig
            self._cache_ds[index] = downscaled

        orig = self._cache[index]
        downscaled = self._cache_ds[index]
        dw, dh = downscaled.size

        # Create a randomly-cropped, -flipped, and -rotated tile for training
        if self._training:
            rnd_y = random.randint(0, dh - self._downscaled_size)
            rnd_x = random.randint(0, dw - self._downscaled_size)
            downscaled = downscaled.crop(
                (rnd_x, rnd_y,
                 rnd_x + self._downscaled_size, rnd_y + self._downscaled_size))

            rnd_x *= self._scale
            rnd_y *= self._scale
            orig = orig.crop(
                (rnd_x, rnd_y,
                 rnd_x + self._tile_size, rnd_y + self._tile_size))

            if random.random() < 0.5:
                downscaled = ImageOps.mirror(downscaled)
                orig = ImageOps.mirror(orig)

            if random.random() < 0.5:
                downscaled = ImageOps.flip(downscaled)
                orig = ImageOps.flip(orig)

            if random.random() < 0.5:
                downscaled = downscaled.rotate(90)
                orig = orig.rotate(90)

        if vis:
            return downscaled, orig

        # pil -> numpy, HWC -> CHW
        orig = np.transpose(
            np.asarray(orig), (2, 0, 1)).astype(np.float32) / 255.
        downscaled = np.transpose(
            np.asarray(downscaled), (2, 0, 1)).astype(np.float32) / 255.

        return downscaled, orig
