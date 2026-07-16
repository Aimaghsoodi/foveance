#!/usr/bin/env python3
"""Render assets/social-card.png — the 1280x640 image GitHub/LinkedIn/X/Slack show when the repo
link is shared. Clean, high-contrast, on-brand. Requires Pillow: pip install pillow."""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "social-card.png"))

W, H = 1280, 640
INK = (240, 243, 250)
DIM = (150, 158, 184)
FAINT = (95, 104, 134)
AMBER = (245, 158, 11)
INDIGO = (129, 140, 248)
GREEN = (86, 214, 140)
CARD = (24, 29, 52)
MARGIN = 92

FONTDIR = r"C:\Windows\Fonts"


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


def w_of(d, text, fnt):
    return d.textlength(text, font=fnt)


def background():
    """Vertical gradient with a soft amber+indigo radial glow behind the headline."""
    top, bot = (10, 13, 24), (17, 22, 40)
    base = Image.new("RGB", (W, H))
    px = base.load()
    for y in range(H):
        t = y / H
        px_row = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = px_row
    # glow layer
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-160, 120, 620, 760], fill=(245, 158, 11, 34))     # amber, lower-left
    gd.ellipse([760, -220, 1500, 440], fill=(99, 102, 241, 40))    # indigo, upper-right
    glow = glow.filter(ImageFilter.GaussianBlur(150))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    return base.convert("RGB")


def mark(img, cx, cy, r, faint=False):
    """Refined foveation mark: evenly-spaced dot rings fading outward, glowing amber core."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rings = 4
    for ring in range(rings, 0, -1):
        rad = r * ring / rings
        n = 6 * ring
        alpha = int((230 if not faint else 42) * (1.02 - 0.16 * ring))
        dot = max(2, int((r / 26) * (rings - ring + 1.4)))
        col = INDIGO if ring % 2 else (168, 158, 240)
        for k in range(n):
            a = 2 * math.pi * k / n + ring * 0.26
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=col + (alpha,))
    # soft amber core glow
    core = int(r * 0.30)
    ga = 120 if not faint else 30
    d.ellipse([cx - core * 1.9, cy - core * 1.9, cx + core * 1.9, cy + core * 1.9],
              fill=AMBER + (ga,))
    img.paste(Image.alpha_composite(img.convert("RGBA"),
                                    layer.filter(ImageFilter.GaussianBlur(0.4))).convert("RGB"),
              (0, 0))
    if not faint:
        d2 = ImageDraw.Draw(img)
        d2.ellipse([cx - core, cy - core, cx + core, cy + core], fill=AMBER)
        d2.ellipse([cx - core - 6, cy - core - 6, cx + core + 6, cy + core + 6],
                   outline=AMBER, width=3)


def fit(d, text, weight, target_px, max_w):
    fnt = font(target_px, weight)
    while w_of(d, text, fnt) > max_w and target_px > 24:
        target_px -= 2
        fnt = font(target_px, weight)
    return fnt


def main():
    img = background()
    d = ImageDraw.Draw(img)
    usable = W - 2 * MARGIN

    # faint hero mark, upper-right, for depth
    mark(img, 1080, 190, 150, faint=True)
    d = ImageDraw.Draw(img)

    # brand row
    mark(img, MARGIN + 40, 118, 40)
    d = ImageDraw.Draw(img)
    d.text((MARGIN + 100, 92), "Foveance", font=font(46, "semibold"), fill=INK)
    d.text((MARGIN + 102, 148), "LOSSLESS  TOKEN  COMPRESSION", font=font(17, "semibold"),
           fill=FAINT)

    # headline. The number is the measured in-context codec saving (bench/results_replay/
    # codec_compare_summary.csv); "losslessly" is the differentiator, so it carries the accent.
    h1 = fit(d, "75% fewer input tokens.", "black", 88, usable)
    d.text((MARGIN, 244), "75% fewer input tokens.", font=h1, fill=INK)
    y2 = 244 + h1.size + 14
    by = font(88, "black")
    d.text((MARGIN, y2), "Losslessly", font=by, fill=AMBER)
    d.text((MARGIN + w_of(d, "Losslessly", by), y2), ".", font=by, fill=INK)

    # subhead
    d.text((MARGIN, y2 + by.size + 26),
           "A real codec for LLM context. Nothing dropped, every fact kept.",
           font=font(28, "regular"), fill=DIM)

    # bottom row: install pill + repo url on one baseline
    pill_y = H - 108
    cmd = "$ pip install foveance"
    cf = font(27, "mono")
    pw = int(w_of(d, cmd, cf)) + 56
    d.rounded_rectangle([MARGIN, pill_y, MARGIN + pw, pill_y + 56], radius=14,
                        fill=CARD, outline=(52, 60, 96), width=1)
    d.text((MARGIN + 28, pill_y + 13), cmd, font=cf, fill=GREEN)
    url = "github.com/aimaghsoodi/foveance"
    uf = font(25, "semibold")
    d.text((W - MARGIN - w_of(d, url, uf), pill_y + 15), url, font=uf, fill=INDIGO)

    img.save(OUT)
    print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
