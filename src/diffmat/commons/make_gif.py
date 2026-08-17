import sys
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


if len(sys.argv) < 2:
    print("Usage: python make_gif.py <image_folder> [output.gif] [duration] [loop]")
    sys.exit(1)


# Image sequence folder
folder = Path(sys.argv[1]).resolve()

if not folder.is_dir():
    print(f"Error: folder does not exist: {folder}")
    sys.exit(1)


# --------------------------------------------------
# Output file
# --------------------------------------------------

if len(sys.argv) >= 3:
    output_arg = Path(sys.argv[2])

    if output_arg.is_absolute():
        # Absolute path: use exactly as given
        output = output_arg
    else:
        # Filename: save next to the image folder
        output = folder.parent / output_arg
else:
    # No filename: use folder name and save next to folder
    output = folder.parent / f"{folder.name}.gif"


# --------------------------------------------------
# Animation settings
# --------------------------------------------------

duration = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
loop = int(sys.argv[4]) if len(sys.argv) >= 5 else 0


# --------------------------------------------------
# Find image files
# --------------------------------------------------

extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

files = [
    f for f in folder.iterdir()
    if f.is_file() and f.suffix.lower() in extensions
]


# Natural sorting: frame_2 before frame_10
def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


files.sort(key=natural_key)


if not files:
    print(f"Error: no image files found in {folder}")
    sys.exit(1)


# --------------------------------------------------
# Create labelled frames
# --------------------------------------------------

font_size = 32
label_position = (20, 20)

iterations = np.arange(0, 11) * 3

images = []

for i, file in enumerate(files):

    # Open while preserving transparency
    image = Image.open(file).convert("RGBA")

    # Put transparent areas onto a white background
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("RGB")

    draw = ImageDraw.Draw(image)

    label = f"iteration {iterations[i]}"

    # Draw a black outline around the text so it remains
    # readable on both light and dark images.
    x, y = label_position

    draw.text(
        (x, y),
        label,
        font_size=font_size,
        fill="white",
        stroke_width=10,
        stroke_fill="black",
    )

    images.append(image)


# --------------------------------------------------
# Create GIF
# --------------------------------------------------

print(f"Input folder : {folder}")
print(f"Frames       : {len(files)}")
print(f"Output       : {output}")
print(f"Duration     : {duration} ms/frame")
print(f"Loop         : {loop}")

images[0].save(
    output,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=loop,
)

print("Done.")
