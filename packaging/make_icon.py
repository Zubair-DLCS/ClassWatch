"""Draws the ClassWatch icon (an eye on a blue tile) and saves classwatch.ico / .png."""
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 40, S - 40, S - 40], radius=210, fill=(47, 109, 181))
# almond-shaped eye: the overlap of two circles
W, H = 800, 440
R = ((W / 2) ** 2 + (H / 2) ** 2) / H
off = R - H / 2
c0 = S / 2
eye = Image.new("L", (S, S), 0)
ImageDraw.Draw(eye).ellipse([c0 - R, c0 + off - R, c0 + R, c0 + off + R], fill=255)
lower = Image.new("L", (S, S), 0)
ImageDraw.Draw(lower).ellipse([c0 - R, c0 - off - R, c0 + R, c0 - off + R], fill=255)
eye = Image.composite(eye, Image.new("L", (S, S), 0), lower)
img.paste((255, 255, 255, 255), (0, 0), eye)
c = S / 2
d.ellipse([c - 165, c - 165, c + 165, c + 165], fill=(32, 48, 72))
d.ellipse([c - 95, c - 95, c + 95, c + 95], fill=(120, 220, 120))
d.ellipse([c - 40, c - 40, c + 40, c + 40], fill=(32, 48, 72))
d.ellipse([c + 45, c - 120, c + 105, c - 60], fill=(255, 255, 255))
out = Path(__file__).parent
img.save(out / "classwatch.png")
img.save(out / "classwatch.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved", out / "classwatch.ico")
