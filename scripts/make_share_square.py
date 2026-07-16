#!/usr/bin/env python3
"""Render assets/share-square.png (1080x1080) for LinkedIn/Instagram image posts.
Same brand system as the social card. Requires Pillow."""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "share-square.png"))
FONTDIR = r"C:\Windows\Fonts"

S = 1080
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
        for path in (os.path.join(FONTDIR, f), f):
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
    return ImageFont.load_default()


def background():
    top, bot = (10, 13, 24), (17, 22, 40)
    base = Image.new("RGB", (S, S))
    px = base.load()
    for y in range(S):
        t = y / S
        row = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        for x in range(S):
            px[x, y] = row
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-200, 500, 560, 1260], fill=(245, 158, 11, 30))
    gd.ellipse([620, -260, 1360, 480], fill=(99, 102, 241, 40))
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    return Image.alpha_composite(base.convert("RGBA"), glow).convert("RGB")


def mark(img, cx, cy, r, faint=False):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rings = 4
    for ring in range(rings, 0, -1):
        rad = r * ring / rings
        n = 6 * ring
        alpha = int((230 if not faint else 40) * (1.02 - 0.16 * ring))
        dot = max(2, int((r / 26) * (rings - ring + 1.4)))
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
        d2.ellipse([cx - core - 7, cy - core - 7, cx + core + 7, cy + core + 7],
                   outline=AMBER, width=3)


def center(d, text, fnt, y, fill):
    w = d.textlength(text, font=fnt)
    d.text(((S - w) / 2, y), text, font=fnt, fill=fill)


def main():
    img = background()
    mark(img, S // 2 + 360, 250, 150, faint=True)
    mark(img, S // 2, 250, 66)
    d = ImageDraw.Draw(img)

    center(d, "Foveance", font(58, "semibold"), 340, INK)
    center(d, "LOSSLESS  TOKEN  COMPRESSION", font(20, "semibold"), 410, FAINT)

    center(d, "75% fewer tokens.", font(94, "black"), 500, INK)
    # "Losslessly." on its own line with the amber accent (the whole line fits the square)
    f = font(94, "black")
    a, b = "", "Losslessly."
    wa, wb = d.textlength(a, font=f), d.textlength(b, font=f)
    x0 = (S - (wa + wb)) / 2
    d.text((x0, 604), a, font=f, fill=INK)
    d.text((x0 + wa, 604), b, font=f, fill=AMBER)

    center(d, "Same code. Same answers. Fewer tokens.", font(34, "regular"), 742, DIM)

    cmd = "$ pip install foveance"
    cf = font(32, "mono")
    pw = int(d.textlength(cmd, font=cf)) + 64
    px0 = (S - pw) / 2
    d.rounded_rectangle([px0, 852, px0 + pw, 852 + 66], radius=16, fill=CARD, outline=(52, 60, 96))
    d.text((px0 + 32, 852 + 15), cmd, font=cf, fill=GREEN)

    center(d, "github.com/aimaghsoodi/foveance", font(28, "semibold"), 968, INDIGO)

    img.save(OUT)
    print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
