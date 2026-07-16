#!/usr/bin/env python3
"""Render assets/twitter-header.png (1500x500) for the X/Twitter profile banner.
Content is centered and kept out of the bottom-left (avatar) safe zone. Requires Pillow."""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "twitter-header.png"))
FONTDIR = r"C:\Windows\Fonts"

W, Hh = 1500, 500
INK = (240, 243, 250)
DIM = (156, 164, 190)
FAINT = (95, 104, 134)
AMBER = (245, 158, 11)
INDIGO = (129, 140, 248)
GREEN = (86, 214, 140)
CARD = (24, 29, 52)


def font(px, weight="regular"):
    files = {
        "black": ["seguibl.ttf", "ariblk.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
        "semibold": ["seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
        "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
        "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"],
    }[weight]
    for f in files:
        for p in (os.path.join(FONTDIR, f), f):
            try:
                return ImageFont.truetype(p, px)
            except OSError:
                continue
    return ImageFont.load_default()


def background():
    top, bot = (10, 13, 24), (17, 22, 40)
    base = Image.new("RGB", (W, Hh))
    px = base.load()
    for y in range(Hh):
        t = y / Hh
        row = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = row
    glow = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([200, -260, 900, 360], fill=(245, 158, 11, 26))
    gd.ellipse([850, 60, 1600, 760], fill=(99, 102, 241, 40))
    glow = glow.filter(ImageFilter.GaussianBlur(150))
    return Image.alpha_composite(base.convert("RGBA"), glow).convert("RGB")


def mark(img, cx, cy, r, faint=False):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ring in range(4, 0, -1):
        rad = r * ring / 4
        n = 6 * ring
        alpha = int((230 if not faint else 40) * (1.02 - 0.16 * ring))
        dot = max(2, int((r / 26) * (4 - ring + 1.4)))
        col = INDIGO if ring % 2 else (168, 158, 240)
        for k in range(n):
            a = 2 * math.pi * k / n + ring * 0.26
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=col + (alpha,))
    core = int(r * 0.30)
    d.ellipse([cx - core * 1.9, cy - core * 1.9, cx + core * 1.9, cy + core * 1.9],
              fill=AMBER + (120 if not faint else 26,))
    img.paste(Image.alpha_composite(img.convert("RGBA"),
                                    layer.filter(ImageFilter.GaussianBlur(0.4))).convert("RGB"),
              (0, 0))
    if not faint:
        d2 = ImageDraw.Draw(img)
        d2.ellipse([cx - core, cy - core, cx + core, cy + core], fill=AMBER)
        d2.ellipse([cx - core - 6, cy - core - 6, cx + core + 6, cy + core + 6],
                   outline=AMBER, width=3)


def center(d, text, fnt, y, fill, cx=W // 2):
    w = d.textlength(text, font=fnt)
    d.text((cx - w / 2, y), text, font=fnt, fill=fill)


def main():
    img = background()
    mark(img, 1300, 150, 130, faint=True)
    d = ImageDraw.Draw(img)

    # brand
    mark(img, W // 2 - 250, 92, 34)
    d = ImageDraw.Draw(img)
    d.text((W // 2 - 210, 70), "Foveance", font=font(40, "semibold"), fill=INK)

    # headline (centered, one line) with amber accent
    f = font(72, "black")
    a, b = "75% fewer input tokens. ", "Losslessly."
    wa, wb = d.textlength(a, font=f), d.textlength(b, font=f)
    x0 = (W - (wa + wb)) / 2
    d.text((x0, 160), a, font=f, fill=INK)
    d.text((x0 + wa, 160), b, font=f, fill=AMBER)

    center(d, "Same code. Same answers. Fewer tokens.  ·  Open source.", font(28, "regular"),
           260, DIM)

    cmd = "$ pip install foveance"
    cf = font(28, "mono")
    pw = int(d.textlength(cmd, font=cf)) + 56
    px0 = (W - pw) / 2
    d.rounded_rectangle([px0, 330, px0 + pw, 330 + 58], radius=14, fill=CARD, outline=(52, 60, 96))
    d.text((px0 + 28, 330 + 14), cmd, font=cf, fill=GREEN)
    center(d, "github.com/aimaghsoodi/foveance", font(24, "semibold"), 416, INDIGO)

    img.save(OUT)
    print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
