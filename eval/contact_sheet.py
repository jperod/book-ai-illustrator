"""Contact sheets for a golden run: tiles of generated images, each labelled with its case, seed and
arm, so a run can be reviewed in a few views. Needs Pillow, which is not a project dependency:

    uv run --with pillow python eval/contact_sheet.py eval/runs/<run-dir> [--per-sheet 6]
    uv run --with pillow python eval/contact_sheet.py eval/runs/<sweep-dir> --bottom --per-sheet 12

Writes sheet_01.png, sheet_02.png, ... into the run folder and prints the order of the tiles.
--bottom shows only the bottom 30% of each picture, where garbled lettering appears, so a seed sweep
can be checked for lettering in a few tall-to-short views. It writes bottom_sheet_NN.png.
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from common import load_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--bottom", action="store_true", help="show only the bottom 30%% of each picture")
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    cols = 3
    label_h = 28
    tile_h = int(args.tile * (0.3 if args.bottom else 1.0))
    prefix = "bottom_sheet" if args.bottom else "sheet"
    font = ImageFont.load_default()
    for sheet_no, start in enumerate(range(0, len(rows), args.per_sheet), 1):
        chunk = rows[start : start + args.per_sheet]
        n_rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * args.tile, n_rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        print(f"{prefix} {sheet_no:02d}:")
        for i, row in enumerate(chunk):
            x = (i % cols) * args.tile
            y = (i // cols) * (tile_h + label_h)
            img = Image.open(os.path.join(args.run_dir, row["image"])).convert("RGB")
            if args.bottom:
                w, h = img.size
                img = img.crop((0, int(h * 0.7), w, h))
            sheet.paste(img.resize((args.tile, tile_h)), (x, y + label_h))
            label = f"{i + 1}. {row['case']}  seed {row['seed']}  [{row['arm']}]"
            draw.text((x + 6, y + 8), label, fill="black", font=font)
            print(f"  tile {i + 1}: {row['case']} seed {row['seed']}")
        out = os.path.join(args.run_dir, f"{prefix}_{sheet_no:02d}.png")
        sheet.save(out)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
