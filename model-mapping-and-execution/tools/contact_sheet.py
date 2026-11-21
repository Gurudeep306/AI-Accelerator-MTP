#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a contact sheet of real dataset frames with their labels on.

    python3 tools/contact_sheet.py <folder>

The folder should hold images and, beside each one, a label file of the same
name with a .txt extension, one line per object in the normalised format

    class  x_centre  y_centre  width  height

which is what the detector's own training code reads. Both the usual layouts
work: images and labels side by side in one folder, or the images/ and labels/
pair the training code uses.

The simulation does not need any of this. A cycle count depends on the shape of
a tensor and not on what is inside it. This exists only so that a demonstration
can put real frames on screen while the letterboxing is being explained.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src'))
from common_paths import OUT, load

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX = 4
TARGET = 640          # the square the detector is fed


def find_label(img):
    cands = [img.with_suffix('.txt')]
    if img.parent.name == 'images':
        cands.append(img.parent.parent / 'labels' / (img.stem + '.txt'))
    cands.append(img.parent / 'labels' / (img.stem + '.txt'))
    for c in cands:
        if c.exists():
            return c
    return None


def read_label(p):
    out = []
    if p is None:
        return out
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            out.append((int(float(f[0])), *(float(x) for x in f[1:5])))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print('This helper needs Pillow:  pip install pillow')
        return 1

    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f'not a folder: {root}')
        return 1

    imgs = sorted(p for p in root.rglob('*') if p.suffix.lower() in EXT)[:MAX]
    if not imgs:
        print(f'no images found under {root}')
        return 1

    names = load('measured_accuracy.json')['per_class']
    names = [k for k in names if not k.startswith('_')]

    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 13)
        small = ImageFont.truetype('DejaVuSans.ttf', 11)
    except Exception:
        font = small = ImageFont.load_default()

    CW, pad, cap = 420, 12, 34
    tiles = []
    for p in imgs:
        im = Image.open(p).convert('RGB')
        w0, h0 = im.size
        lab = read_label(find_label(p))
        sc = CW / w0
        im = im.resize((CW, int(h0 * sc)))
        d = ImageDraw.Draw(im)
        for c, x, y, w, h in lab:
            x1, y1 = (x - w / 2) * im.width, (y - h / 2) * im.height
            x2, y2 = (x + w / 2) * im.width, (y + h / 2) * im.height
            d.rectangle([x1, y1, x2, y2], outline=(176, 57, 46), width=2)
            if 0 <= c < len(names):
                d.text((x1 + 2, max(0, y1 - 12)), names[c], fill=(176, 57, 46),
                       font=small)
        # the letterbox the detector would apply
        s = TARGET / max(w0, h0)
        padpx = int(round((TARGET - min(w0, h0) * s) / 2))
        tiles.append((im, p.name, w0, h0, len(lab), padpx))

    cols = 2
    rows = -(-len(tiles) // cols)
    th = max(t[0].height for t in tiles)
    W = cols * CW + (cols + 1) * pad
    H = rows * (th + cap) + (rows + 1) * pad + 46
    sheet = Image.new('RGB', (W, H), (251, 248, 239))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 12), 'Frames from the dataset the detector was trained on, '
                       'with their labels drawn on', fill=(30, 58, 95), font=font)
    dr.text((pad, 29), f'Each frame is letterboxed to {TARGET}x{TARGET} before '
                       f'the detector sees it. The grey bars carry no picture.',
            fill=(90, 107, 122), font=small)

    for i, (im, name, w0, h0, n, padpx) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (CW + pad)
        y = 46 + pad + r * (th + cap + pad)
        sheet.paste(im, (x, y))
        share = 100.0 * (2 * padpx * TARGET) / (TARGET * TARGET)
        dr.text((x, y + im.height + 4), f'{name}   {w0}x{h0}   {n} objects',
                fill=(30, 58, 95), font=small)
        dr.text((x, y + im.height + 18),
                f'letterboxed to {TARGET}x{TARGET}: {padpx} px grey above and '
                f'below, {share:.1f}% of the input', fill=(90, 107, 122),
                font=small)

    dest = OUT / 'dataset_sheet.png'
    sheet.save(dest)
    print(f'wrote {dest}')
    print(f'{len(tiles)} frames, '
          f'{sum(t[4] for t in tiles)} labelled objects between them')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
