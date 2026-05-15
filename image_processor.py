#!/usr/bin/env python3
# Safe-only pipeline:
# Photos: supported type -> blur check -> min resolution -> size <= 200 KB
# Signatures: supported type -> DPI check -> ink extraction -> width/aspect check -> connected components -> size <= 100 KB

from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image


HEIC_EXTS = {".heic", ".heif"}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"} | HEIC_EXTS

PHOTO_MAX_KB = 200
SIG_MAX_KB = 100
PHOTO_MIN_W, PHOTO_MIN_H = 250, 300
PHOTO_TARGET_LONG_SIDE = 800
FACE_DETECT_LONG_SIDE = 1000
PHOTO_FACE_MIN_AREA = 0.04
PHOTO_FACE_MAX_AREA = 0.65
PHOTO_FACE_CENTER_TOLERANCE = 0.20
SIG_MIN_W = 200
SIGNATURE_FACE_MIN_AREA = 0.02

_FACE_CASCADE = None


def valid_photo_aspect_ratio(width, height):
    ratio = width / height
    return 0.65 <= ratio <= 0.85 or 0.95 <= ratio <= 1.05


def face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            raise ValueError("Face detector unavailable")
    return _FACE_CASCADE


def _read_heic(p):
    """Read a HEIC/HEIF image via pillow-heif and return as BGR numpy array."""
    try:
        from pillow_heif import open_heif
        heif_file = open_heif(str(p))
        pil_img = heif_file.to_pillow()
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Unreadable HEIC image")


def read_img(p):
    if p.suffix.lower() in HEIC_EXTS:
        return _read_heic(p)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Unreadable image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        color = img[:, :, :3]
        alpha = img[:, :, 3] / 255.0
        white = np.full_like(color, 255)
        img = (color * alpha[:, :, None] + white * (1 - alpha[:, :, None])).astype(np.uint8)
    return img


def check_signature_dpi(path):
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if dpi is None:
                return
            if isinstance(dpi, tuple):
                xdpi, ydpi = dpi
            else:
                xdpi = ydpi = dpi

            if xdpi < 72 or ydpi < 72:
                raise ValueError(
                    f"Signature DPI too low ({xdpi}x{ydpi})"
                )
    except OSError:
        # Many mobile images omit DPI metadata.
        # Treat missing DPI metadata as acceptable if resolution is sufficient.
        pass


def save_jpg_under_limit(img, out_path, max_kb, min_w=1, min_h=1):
    working = img
    while True:
        for q in range(92, 34, -3):
            ok, buf = cv2.imencode(
                ".jpg",
                working,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    q,
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    1,
                ],
            )
            if ok and len(buf) <= max_kb * 1024:
                out_path.write_bytes(buf.tobytes())
                return

        h, w = working.shape[:2]
        next_w = max(min_w, int(w * 0.9))
        next_h = max(min_h, int(h * 0.9))
        if (next_w, next_h) == (w, h):
            break
        working = cv2.resize(working, (next_w, next_h), interpolation=cv2.INTER_AREA)
    raise ValueError("Could not compress under size limit")


def save_png_under_limit(img, out_path, max_kb, min_w=1):
    working = img
    while True:
        ok, buf = cv2.imencode(".png", working, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if ok and len(buf) <= max_kb * 1024:
            out_path.write_bytes(buf.tobytes())
            return

        h, w = working.shape[:2]
        next_w = max(min_w, int(w * 0.9))
        if next_w == w:
            break
        next_h = max(1, int(h * (next_w / w)))
        working = cv2.resize(working, (next_w, next_h), interpolation=cv2.INTER_AREA)
    raise ValueError("Could not compress under size limit")


def blur_score(img):
    h, w = img.shape[:2]
    long_side = max(w, h)
    if long_side > PHOTO_TARGET_LONG_SIDE:
        scale = PHOTO_TARGET_LONG_SIDE / long_side
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def resize_for_detection(img):
    h, w = img.shape[:2]
    long_side = max(w, h)
    if long_side <= FACE_DETECT_LONG_SIDE:
        return img, 1.0
    scale = FACE_DETECT_LONG_SIDE / long_side
    resized = cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def detect_faces(img, min_area=0.0, max_area=1.0):
    small, scale = resize_for_detection(img)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_size = max(30, int(min(small.shape[:2]) * 0.05))
    raw_faces = face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )

    image_area = small.shape[0] * small.shape[1]
    faces = []
    for x, y, w, h in raw_faces:
        area_fraction = (w * h) / image_area
        if min_area <= area_fraction <= max_area:
            faces.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
    return faces


def ensure_plain_light_photo_background(img, face):
    small, scale = resize_for_detection(img)
    x, y, w, h = [int(v * scale) for v in face]
    H, W = small.shape[:2]

    mask = np.ones((H, W), dtype=np.uint8) * 255
    pad_x = int(w * 0.75)
    pad_y = int(h * 1.00)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    mask[y0:y1, x0:x1] = 0

    margin = max(10, int(min(W, H) * 0.12))
    border = np.zeros((H, W), dtype=np.uint8)
    border[:margin, :] = 255
    border[-margin:, :] = 255
    border[:, :margin] = 255
    border[:, -margin:] = 255
    sample_mask = cv2.bitwise_and(mask, border)

    if cv2.countNonZero(sample_mask) < 1000:
        sample_mask = border
    if cv2.countNonZero(sample_mask) < 1000:
        raise ValueError("Could not validate photo background")

    sample = small[sample_mask > 0]
    hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    light_plain_fraction = np.mean((value >= 150) & (saturation <= 90))

    if value.mean() < 135 or saturation.mean() > 85 or value.std() > 75:
        raise ValueError("Photo background must be plain white or light")
    if light_plain_fraction < 0.35:
        raise ValueError("Photo background must be plain white or light")


def ensure_photo_face_constraints(img):
    faces = detect_faces(img, PHOTO_FACE_MIN_AREA, PHOTO_FACE_MAX_AREA)
    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face, detected {len(faces)}")

    x, y, w, h = faces[0]
    H, W = img.shape[:2]
    face_area = (w * h) / (W * H)
    if not (PHOTO_FACE_MIN_AREA <= face_area <= PHOTO_FACE_MAX_AREA):
        raise ValueError("Photo face size is outside allowed range")

    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    if abs(cx - 0.5) > PHOTO_FACE_CENTER_TOLERANCE:
        raise ValueError("Photo face is not horizontally centered")
    if abs(cy - 0.5) > PHOTO_FACE_CENTER_TOLERANCE:
        raise ValueError("Photo face is not vertically centered")

    ensure_plain_light_photo_background(img, faces[0])


def ensure_no_signature_face(img):
    faces = detect_faces(img, SIGNATURE_FACE_MIN_AREA, 1.0)
    if faces:
        raise ValueError("Signature appears to contain a face")


def ensure_signature_background(img, ink_mask):
    background = img[ink_mask == 0]
    if len(background) < 1000:
        raise ValueError("Could not validate signature background")

    hsv = cv2.cvtColor(background.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    white_fraction = np.mean((value >= 180) & (saturation <= 60))

    if value.mean() < 170 or saturation.mean() > 70 or value.std() > 65:
        raise ValueError("Signature background must be white or transparent")
    if white_fraction < 0.70:
        raise ValueError("Signature background must be white or transparent")


def ensure_photo(img):
    h, w = img.shape[:2]
    if not valid_photo_aspect_ratio(w, h):
        raise ValueError("Bad photo aspect ratio")

    # Constraint: reject blur/out-of-focus images
    if blur_score(img) < 80:
        raise ValueError("Photo too blurry")

    ensure_photo_face_constraints(img)

    # Constraint: minimum resolution 250x300
    scale = max(PHOTO_MIN_W / w, PHOTO_MIN_H / h, 1.0)

    if scale > 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )
        h, w = img.shape[:2]

    long_side = max(w, h)
    if long_side > PHOTO_TARGET_LONG_SIDE:
        scale = PHOTO_TARGET_LONG_SIDE / long_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return img


def ensure_signature(img, path):
    # Constraint: minimum 72 DPI recommended
    check_signature_dpi(path)
    ensure_no_signature_face(img)

    # Constraint: white/transparent background + ink analysis + reject blank
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ensure_signature_background(img, bw)

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = nlab - 1  # ignore background

    if comps == 0:
        raise ValueError("Blank signature")
    if comps > 50:
        raise ValueError("Signature too noisy")

    pts = cv2.findNonZero(bw)
    if pts is None:
        raise ValueError("Blank signature")

    x, y, w, h = cv2.boundingRect(pts)

    # Constraint: width at least 200 px
    if w < SIG_MIN_W:
        scale = SIG_MIN_W / w
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        pts = cv2.findNonZero(bw)
        x, y, w, h = cv2.boundingRect(pts)

    # Constraint: rectangular aspect ratio 1.5 to 4.5
    ar = w / h
    if not (1.5 <= ar <= 4.5):
        raise ValueError("Bad signature aspect ratio")

    # Tight crop to ink
    pad = 10
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

    cropped = bw[y0:y1, x0:x1]
    clean = np.full(cropped.shape, 255, dtype=np.uint8)
    clean[cropped > 0] = 0
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def process_dir(subdir, mode):
    d = Path(subdir)
    if not d.exists():
        return
    for p in d.iterdir():
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        if p.stem.endswith("_processed"):
            continue
        try:
            img = read_img(p)
            if mode == "photo":
                out = ensure_photo(img)
                out_path = p.with_name(f"{p.stem}_processed.jpg")
                save_jpg_under_limit(out, out_path, PHOTO_MAX_KB, PHOTO_MIN_W, PHOTO_MIN_H)
            else:
                out = ensure_signature(img, p)
                out_path = p.with_name(f"{p.stem}_processed.png")
                save_png_under_limit(out, out_path, SIG_MAX_KB, SIG_MIN_W)
            print(f"OK  {p.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERR {p.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Folder containing photos/ and signatures/")
    args = ap.parse_args()
    root = Path(args.root)
    process_dir(root / "photos", "photo")
    process_dir(root / "signatures", "signature")


if __name__ == "__main__":
    main()
