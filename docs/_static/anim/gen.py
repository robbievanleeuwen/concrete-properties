"""Script that combines multiple pngs into a gif."""

import glob

from PIL import Image

# filepaths
fp_in = "*.png"
fp_out = "anim.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(
    fp=fp_out,
    format="GIF",
    append_images=imgs,
    save_all=True,
    duration=1000 / 24,
    loop=0,
)
