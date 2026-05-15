# UUCMS Photo Adherance and Verification Helper

This project processes your passport photos and signatures into upload-friendly files for government websites like https://uucms.karnataka.gov.in with stringent checks 

UUCMS has the following flimsy requirements (possibly had some "overly enthusiastic" interns)

```
Photo Requirements:

Allowed file types: JPG, JPEG, PNG.
Maximum Photo file size: 200 KB.
Please upload a passport-size photo and signature. The same photo will be used for your hall ticket, marks card, and convocation certificate.
Minimum resolution for photo: 250 X 300 pixels (for print clarity).
Aspect ratio must follow standard passport dimensions:
0.65 to 0.85: Standard 35x45mm (UK/EU/India)(UK/EU/India)
0.95 to 1.05: Standard 2x2 inch
Photo Quality Checks
Blur detection:Rejects out-of-focus images
Background:Rejects busy backgrounds (requires a plain white or light background)
Face Detection: Exactly one face must be detected
Centering: Face must be horizontally and vertically centered
Sizing: Face must occupy between 4% and 65% of the total image area (prevents selfies or too-distant shots)
Liveness/Authenticity:Rejects avatars, cartoons, illustrations, and hyper-realistic AI-generated faces using texture and color-richness analysis

Signature Requirements:

File Size: Maximum 100 KB
DPI: Minimum 72 DPI recommended for mobile-captured signatures.
Resolution: Width must be at least 200 pixels.
Aspect Ratio: Must be rectangular, between 1.5 and 4.5
Background: Must be white paper or transparent. Physically dark/noisy backgrounds are rejected.
Portrait Check:Rejects images if a face is detected (prevents uploading portraits as signatures).
Ink Analysis:
Detects ink strokes (connected components)
Rejects blank images.
Rejects signs with too much noise (detected as >50 disconnected strokes)
```

It reads input files from:

- `photos/`
- `signatures/`

It writes processed files next to the originals using `_processed` in the filename.

Examples:

- `photos/IMG_3348.HEIC` -> `photos/IMG_3348_processed.jpg`
- `signatures/Signature.jpg` -> `signatures/Signature_processed.png`

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Run the processor:

```bash
uv run python image_processor.py --root .
```

If your `photos/` and `signatures/` folders are inside another directory, pass that directory as `--root`:

```bash
uv run python image_processor.py --root /path/to/folder
```

## Input Folders

Expected project layout:

```text
photo-processor/
  image_processor.py
  photos/
    your-photo.jpg
    your-photo.png
    your-photo.heic
  signatures/
    your-signature.jpg
    your-signature.png
```

Already processed files ending in `_processed` are skipped so the script does not keep re-processing its own output.

## Definitively Enforced Constraints

The following constraints are enforced by the current script when it successfully produces output.

### Photo

| Constraint | Enforced? | How |
| --- | --- | --- |
| Input file is readable as an image | Yes | OpenCV or Pillow HEIF decode must succeed. |
| Input extension is supported | Yes | Accepts `.jpg`, `.jpeg`, `.png`, `.heic`, and `.heif`. |
| Output file type | Yes | Photos are always written as `.jpg`. |
| Maximum output file size: 200 KB | Yes | JPEG quality and dimensions are reduced until the file is under 200 KB, while preserving minimum resolution. |
| Minimum output resolution: 250 x 300 px | Yes | Small images are upscaled; compression will not reduce below this minimum. |
| Passport aspect ratio | Yes | Original image width/height must be `0.65` to `0.85`, or `0.95` to `1.05`. |
| Blur rejection | Yes, heuristic | Uses Laplacian variance after normalizing large images to passport scale. Images scoring below `80` are rejected. |
| Background must be plain white or light | Yes, heuristic | Samples border/background regions outside the detected face and rejects dark, saturated, or visually busy backgrounds. |
| Exactly one face must be detected | Yes, heuristic | Uses OpenCV Haar face detection and counts only faces in the allowed passport-size area range. |
| Face must be horizontally and vertically centered | Yes | The detected face center must be within `20%` of the image center on both axes. |
| Face must occupy 4% to 65% of image area | Yes | The detected face bounding box must be within this area range. |

### Signature

| Constraint | Enforced? | How |
| --- | --- | --- |
| Input file is readable as an image | Yes | OpenCV or Pillow HEIF decode must succeed. |
| Input extension is supported | Yes | Accepts `.jpg`, `.jpeg`, `.png`, `.heic`, and `.heif`. |
| Output file type | Yes | Signatures are always written as `.png`. |
| Maximum output file size: 100 KB | Yes | PNG compression is applied; if needed, dimensions are reduced while keeping width at least 200 px. |
| Minimum width: 200 px | Yes | Signature ink bounding box is upscaled if needed. |
| Aspect ratio: 1.5 to 4.5 | Yes | Checked against the detected ink bounding box. |
| Blank signature rejection | Yes | Rejects images with no detected ink pixels. |
| Too many disconnected strokes/noise | Yes | Rejects signatures with more than 50 connected ink components. |
| White or transparent input background | Yes, heuristic | Transparent pixels are composited onto white; detected background must be bright, low-saturation, and not too noisy. |
| White output background | Yes | The processed output is cleaned to black ink on a white background. |
| Minimum DPI: 72, when metadata exists | Yes | Images with explicit DPI below 72 are rejected. Images with missing DPI metadata are accepted. |
| Reject portraits/faces in signature uploads | Yes, heuristic | Uses OpenCV Haar face detection and rejects signature images with a detected portrait-sized face. |
| Reject physical dark/noisy backgrounds | Yes, heuristic | Validates the detected non-ink background for brightness, low saturation, and limited variation. |

## Notes

- HEIC/HEIF input is accepted for convenience, even though some official portals only allow JPG, JPEG, and PNG uploads. The generated photo output is JPG and the generated signature output is PNG.
- Blur, background, and face checks are local OpenCV heuristics, not the target portal's private validator.
- The script prepares files for the listed upload constraints, but the target portal may still apply stricter checks.
