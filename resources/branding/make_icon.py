# Build neoMyelin_icon.png: white mark centered on a dark indigo square.
# Usage: python make_icon.py  (run from resources/branding)
from PIL import Image

full = Image.open("neoMyelin.png").convert("RGBA")
mark = full.crop((0, 0, 228, 220))  # bbox of the block mark in the 1400px render

# Brand background: deep indigo/slate (matches the default dark theme).
SIZE = 256
bg = Image.new("RGBA", (SIZE, SIZE), (22, 27, 46, 255))

# Fit the mark into the square with padding, preserving aspect.
pad = 28
max_w = SIZE - 2 * pad
max_h = SIZE - 2 * pad
ratio = min(max_w / mark.width, max_h / mark.height)
w = round(mark.width * ratio)
h = round(mark.height * ratio)
mark_resized = mark.resize((w, h), Image.LANCZOS)

x = (SIZE - w) // 2
y = (SIZE - h) // 2
bg.alpha_composite(mark_resized, (x, y))

bg.save("neoMyelin_icon.png")
print("saved neoMyelin_icon.png", bg.size)
