"""Turn the label-set photos into a PDF for hand reading, plus a blank text file to fill in.

    python labelling/make_sheet.py <folder with photos.zip and photos.csv>
      -> <folder>/plates_to_label.pdf, <folder>/labels.txt

Plates show real registrations: keep the folder out of the repository.
"""
import csv
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

COLS, ROWS = 2, 3
BOX_W, BOX_H = 8.4 * cm, 7.0 * cm

folder = Path(sys.argv[1])
ids = [r["id"] for r in csv.DictReader(open(folder / "photos.csv", encoding="utf-8"))]
zf = zipfile.ZipFile(folder / "photos.zip")

pdf = canvas.Canvas(str(folder / "plates_to_label.pdf"), pagesize=A4)
W, H = A4
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(2 * cm, H - 3 * cm, f"Number plates to label ({len(ids)} photos)")
pdf.setFont("Helvetica", 11)
for i, line in enumerate([
        "For each photo, write what its number plate says in labels.txt, on the line with its ID.",
        "If a photo shows more than one plate, label the largest one.",
        "Write every character in reading order (top row, then bottom row),",
        "with no spaces, dots or hyphens. Include province words as printed.",
        "    Devanagari plates in Devanagari, e.g.  P0001  <zone><lot><class><number>",
        "    embossed plates in Latin letters and digits as printed.",
        "If you cannot read every character, write  ?",
        "If the photo has no number plate, write  x",
        "Read what you see; do not guess characters that are not visible.",
]):
    pdf.drawString(2 * cm, H - 4.5 * cm - i * 0.7 * cm, line)
pdf.showPage()

per_page = COLS * ROWS
for start in range(0, len(ids), per_page):
    for n, pid in enumerate(ids[start:start + per_page]):
        col, row = n % COLS, n // COLS
        x = 1.5 * cm + col * (BOX_W + 1.2 * cm)
        y = H - 2 * cm - (row + 1) * (BOX_H + 1.2 * cm)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(x, y + BOX_H + 0.25 * cm, pid)
        im = Image.open(BytesIO(zf.read(f"{pid}.jpg"))).convert("RGB")
        im.thumbnail((700, 700))                      # enough to read; keeps the PDF small
        buf = BytesIO()
        im.save(buf, "JPEG", quality=75)              # reportlab stores a PIL image uncompressed otherwise
        buf.seek(0)
        img = ImageReader(buf)
        iw, ih = img.getSize()
        s = min(BOX_W / iw, BOX_H / ih)
        pdf.drawImage(img, x, y + (BOX_H - ih * s), iw * s, ih * s)
        pdf.rect(x, y, BOX_W, BOX_H, stroke=1, fill=0)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(1.5 * cm, 1 * cm, f"page {start // per_page + 2}")
    pdf.showPage()
pdf.save()

(folder / "labels.txt").write_text("".join(f"{pid}\t\n" for pid in ids), encoding="utf-8")
print(f"wrote {folder / 'plates_to_label.pdf'} ({len(ids)} plates) and {folder / 'labels.txt'}")
