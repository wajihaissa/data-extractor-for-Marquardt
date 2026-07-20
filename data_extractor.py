import argparse
import collections
import csv
import json
import os
import re
import sys
import zipfile
from xml.sax.saxutils import escape as xml_escape

try:
    import fitz
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:
    raise ImportError(
        "Missing required package. Install pymupdf, numpy, pillow. "
        "Example: python -m pip install pymupdf pillow numpy"
    ) from exc


def render_pdf_page(pdf_path, page_index=0, zoom=3.0):
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"PDF page index {page_index} out of range (0..{len(doc)-1})")
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def pil_to_gray(image):
    return np.asarray(image.convert("L"))


def otsu_threshold(gray):
    flat = gray.ravel()
    hist = np.bincount(flat, minlength=256)
    total = flat.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    weight_b = 0.0
    max_var = 0.0
    threshold = 128
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def pad_to_multiple(mask, block_size):
    bh, bw = block_size
    h, w = mask.shape
    pad_h = (-h) % bh
    pad_w = (-w) % bw
    if pad_h == 0 and pad_w == 0:
        return mask
    padded = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    return padded


def block_mask(mask, block_size=(8, 8)):
    bh, bw = block_size
    padded = pad_to_multiple(mask, block_size)
    h, w = padded.shape
    blocks = padded.reshape(h // bh, bh, w // bw, bw)
    any_blocks = np.any(blocks, axis=(1, 3))
    return any_blocks


def connected_components_bool(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components = []
    label = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            queue = collections.deque([(x, y)])
            labels[y, x] = label
            x0 = x
            x1 = x
            y0 = y
            y1 = y
            area = 0
            while queue:
                cx, cy = queue.popleft()
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        queue.append((nx, ny))
            components.append((x0, y0, x1 + 1, y1 + 1, area))
            label += 1
    return components


def detect_plot_regions(image, min_area_ratio=0.01, max_regions=8):
    gray = pil_to_gray(image)
    threshold = otsu_threshold(gray)
    mask = gray < min(200, threshold + 20)
    blocks = block_mask(mask, block_size=(10, 10))
    comps = connected_components_bool(blocks)
    h, w = gray.shape
    min_area = max(2000, int(h * w * min_area_ratio))
    regions = []
    for bx0, by0, bx1, by1, area in comps:
        x0 = bx0 * 10
        y0 = by0 * 10
        x1 = min(bx1 * 10, w)
        y1 = min(by1 * 10, h)
        cw = x1 - x0
        ch = y1 - y0
        region_area = cw * ch
        if region_area < min_area:
            continue
        if cw < 0.15 * w or ch < 0.12 * h:
            continue
        if x0 < 0.02 * w and y0 < 0.02 * h and x1 > 0.98 * w and y1 > 0.98 * h:
            continue
        regions.append((x0, y0, cw, ch, region_area))
    if not regions:
        return [(0, 0, w, h, w * h)]
    regions.sort(key=lambda item: (-item[4], item[1], item[0]))
    selected = []
    for region in regions:
        x, y, cw, ch, area = region
        overlap = False
        for ex, ey, ecw, ech, _ in selected:
            if x < ex + ecw and ex < x + cw and y < ey + ech and ey < y + ch:
                overlap = True
                break
        if not overlap:
            selected.append(region)
        if len(selected) >= max_regions:
            break
    if not selected:
        return [(0, 0, w, h, w * h)]
    return selected


def crop_region(image, region, pad_ratio=0.02):
    x, y, cw, ch, _ = region
    w, h = image.size
    pad_x = int(cw * pad_ratio)
    pad_y = int(ch * pad_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1 - x0, y1 - y0)


def content_bbox(image, thresh=245, pad=20):
    arr = np.asarray(image.convert('L'))
    mask = arr < thresh
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(arr.shape[1], xs.max() + pad + 1)
    y1 = min(arr.shape[0], ys.max() + pad + 1)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def extract_red_markers(image, debug=False):
    """Detect red circular markers (measured data points) on the chart.
    Returns pixel coordinates of marker centers.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    h, w, _ = arr.shape
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    # Red marker detection: high red, low green/blue
    red_score = r - np.maximum(g, b)
    mask = (r > 120) & (red_score > 50) & (g < 150) & (b < 150)
    if not np.any(mask):
        return []
    # Find connected components (marker clusters)
    comps = connected_components_bool(mask)
    points = []
    for x0, y0, x1, y1, area in comps:
        cw = x1 - x0
        ch = y1 - y0
        # Filter for roughly circular shapes (ratio 0.5 to 2.0)
        if area < 20 or area > 5000:
            continue
        ratio = float(cw) / max(1, ch)
        if ratio < 0.5 or ratio > 2.0:
            continue
        # Centroid of the marker
        marker_y, marker_x = np.where(mask[y0:y1, x0:x1])
        cx = x0 + int(np.mean(marker_x))
        cy = y0 + int(np.mean(marker_y))
        points.append((cx, cy))
    return sorted(points, key=lambda p: p[0])


def extract_line_points_color(image, debug=False):
    """Detect colored (blue/cyan) trace by scanning columns for pixels
    where the blue channel is stronger than red/green. Fill small horizontal
    gaps by borrowing nearby columns' detections.
    Coordinates returned are relative to the cropped image (0..width-1).
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    h, w, _ = arr.shape
    top = int(h * 0.05)
    bottom = int(h * 0.95)
    left = int(w * 0.03)
    right = int(w * 0.97)
    sub = arr[top:bottom, left:right]
    r = sub[:, :, 0].astype(int)
    g = sub[:, :, 1].astype(int)
    b = sub[:, :, 2].astype(int)
    score = b - np.maximum(r, g)
    mask = (b > 100) & (score > 30)
    points = []
    counts = mask.sum(axis=0)
    max_gap = 5
    for x in range(mask.shape[1]):
        if counts[x] > 0:
            ys = np.where(mask[:, x])[0]
            weights = b[ys]
            if weights.sum() > 0:
                y = int(np.round(np.average(ys, weights=weights)))
            else:
                y = int(np.median(ys))
            points.append((x + left, y + top))
        else:
            # look for nearby column with detection
            found = False
            for d in range(1, max_gap + 1):
                for nx in (x - d, x + d):
                    if 0 <= nx < mask.shape[1] and counts[nx] > 0:
                        ys = np.where(mask[:, nx])[0]
                        weights = b[ys]
                        if weights.sum() > 0:
                            y = int(np.round(np.average(ys, weights=weights)))
                        else:
                            y = int(np.median(ys))
                        points.append((x + left, y + top))
                        found = True
                        break
                if found:
                    break
    # require a minimum number of color points to trust color extraction
    if len(points) < 8:
        return []
    return points


def extract_line_points(image, debug=False):
    # try red markers first (measured points)
    red_pts = extract_red_markers(image, debug=debug)
    if red_pts and len(red_pts) >= 3:
        return red_pts
    
    # fallback to color-based blue trace
    color_pts = extract_line_points_color(image, debug=debug)
    if color_pts and len(color_pts) >= 8:
        return color_pts

    gray = pil_to_gray(image)
    h, w = gray.shape
    top = int(h * 0.05)
    bottom = int(h * 0.95)
    left = int(w * 0.03)
    right = int(w * 0.97)
    region = gray[top:bottom, left:right]
    threshold = otsu_threshold(region)
    mask = region < min(200, threshold + 20)
    if np.sum(mask) < 5:
        mask = region < 180
    points = []
    for x in range(mask.shape[1]):
        col = mask[:, x]
        if not np.any(col):
            continue
        ys = np.where(col)[0]
        median_y = int(np.median(ys))
        # return coordinates relative to the cropped region
        points.append((x + left, median_y + top))
    if not points:
        mask = gray < min(200, otsu_threshold(gray) + 20)
        for x in range(w):
            col = mask[:, x]
            if not np.any(col):
                continue
            ys = np.where(col)[0]
            median_y = int(np.median(ys))
            points.append((x, median_y))
    return points


def extract_line_points_full(image):
    """Extract line points scanning the full cropped image width (no left/right margins).
    Used as a fallback when the target-x lies outside the initially-extracted x-range.
    """
    gray = pil_to_gray(image)
    h, w = gray.shape
    top = int(h * 0.05)
    bottom = int(h * 0.95)
    region = gray[top:bottom, :]
    threshold = otsu_threshold(region)
    mask = region < min(200, threshold + 20)
    if np.sum(mask) < 5:
        mask = region < 180
    points = []
    for x in range(mask.shape[1]):
        col = mask[:, x]
        if not np.any(col):
            continue
        ys = np.where(col)[0]
        median_y = int(np.median(ys))
        points.append((x, median_y + top))
    if not points:
        mask = gray < min(200, otsu_threshold(gray) + 20)
        for x in range(w):
            col = mask[:, x]
            if not np.any(col):
                continue
            ys = np.where(col)[0]
            median_y = int(np.median(ys))
            points.append((x, median_y))
    return points


def map_points_to_data(pixel_points, region_size, x_min, x_max, y_min, y_max):
    _, _, width, height = region_size
    if width <= 0 or height <= 0:
        return []
    mapped = []
    for px, py in pixel_points:
        x_frac = px / max(1, width - 1)
        y_frac = py / max(1, height - 1)
        x_val = x_min + x_frac * (x_max - x_min)
        y_val = y_max - y_frac * (y_max - y_min)
        mapped.append((x_val, y_val))
    return mapped


def parse_numeric_label(text):
    if not text:
        return None
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_OCR_TEMPLATE_CACHE = None
RENDER_ZOOM = 5.0
MEASUREMENT_LEVELS = [-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
ERROR_LEVELS = [35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35]


def _get_tesseract_command():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _get_tessdata_prefix():
    candidates = [
        os.path.join(os.path.dirname(__file__), "tessdata"),
        os.path.join(os.path.dirname(__file__), "entryPoint", "tessdata"),
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "eng.traineddata")):
            return candidate
    return None


def _build_ocr_templates():
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
    ]
    font_path = next((path for path in font_candidates if os.path.exists(path)), None)
    if font_path is None:
        return {}

    font = ImageFont.truetype(font_path, 42)
    template_chars = "0123456789.-+Vv"
    templates = {}
    for character in template_chars:
        variants = []
        for size in (30, 36, 42, 48):
            for candidate_path in font_candidates:
                if not os.path.exists(candidate_path):
                    continue
                candidate_font = ImageFont.truetype(candidate_path, size)
                canvas = Image.new("L", (72, 72), 255)
                draw = ImageDraw.Draw(canvas)
                bbox = draw.textbbox((0, 0), character, font=candidate_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(((72 - text_width) // 2 - bbox[0], (72 - text_height) // 2 - bbox[1]), character, font=candidate_font, fill=0)
                binary = np.asarray(canvas) < 180
                ys, xs = np.where(binary)
                if xs.size == 0:
                    continue
                x0, x1 = xs.min(), xs.max() + 1
                y0, y1 = ys.min(), ys.max() + 1
                variants.append(binary[y0:y1, x0:x1])
        if variants:
            templates[character] = variants
    return templates


def _get_ocr_templates():
    global _OCR_TEMPLATE_CACHE
    if _OCR_TEMPLATE_CACHE is None:
        _OCR_TEMPLATE_CACHE = _build_ocr_templates()
    return _OCR_TEMPLATE_CACHE


def _normalize_binary_mask(mask, size=(32, 48)):
    if mask.size == 0:
        return None
    image = Image.fromarray((~mask).astype(np.uint8) * 255)
    image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("L", size, 255)
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas) < 128


def _classify_ocr_character(mask):
    templates = _get_ocr_templates()
    if not templates:
        return None

    height, width = mask.shape
    area = int(mask.sum())
    if area == 0:
        return None
    if width <= 4 and height <= 10 and area <= 20:
        return "."
    if width >= 10 and area / max(1, width * height) < 0.12 and width / max(1, height) > 3.0:
        return "-"

    normalized = _normalize_binary_mask(mask)
    if normalized is None:
        return None

    best_character = None
    best_score = -1.0
    for character, template in templates.items():
        for template in template:
            template_image = Image.fromarray((~template).astype(np.uint8) * 255)
            template_image = ImageOps.contain(template_image, (32, 48), method=Image.Resampling.LANCZOS)
            template_canvas = Image.new("L", (32, 48), 255)
            template_canvas.paste(template_image, ((32 - template_image.width) // 2, (48 - template_image.height) // 2))
            template_mask = np.asarray(template_canvas) < 128
            intersection = np.logical_and(normalized, template_mask).sum()
            union = np.logical_or(normalized, template_mask).sum()
            score = intersection / union if union else 0.0
            if score > best_score:
                best_score = score
                best_character = character

    if best_score < 0.2:
        return None
    return best_character


def _simple_numeric_ocr(image):
    if image.mode != "L":
        gray = np.asarray(image.convert("L"))
    else:
        gray = np.asarray(image)

    if gray.size == 0:
        return None

    gray = gray.copy()
    if float(gray.mean()) < 128:
        gray = 255 - gray

    mask = gray < 220
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    mask = mask[y0:y1, x0:x1]

    column_has_ink = mask.any(axis=0)
    segments = []
    start = None
    for index, has_ink in enumerate(column_has_ink):
        if has_ink and start is None:
            start = index
        elif not has_ink and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(column_has_ink)))

    characters = []
    for segment_start, segment_end in segments:
        char_mask = mask[:, segment_start:segment_end]
        if char_mask.size == 0:
            continue
        char_ys, char_xs = np.where(char_mask)
        if char_xs.size == 0:
            continue
        char_mask = char_mask[char_ys.min():char_ys.max() + 1, char_xs.min():char_xs.max() + 1]
        character = _classify_ocr_character(char_mask)
        if character is not None:
            characters.append(character)

    if not characters:
        return None

    text = "".join(characters).replace("vv", "V")
    return text


def ocr_image_to_text(image, whitelist="0123456789.-+eE", psm=7):
    try:
        import pytesseract
    except ImportError:
        return _simple_numeric_ocr(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        red_mask = (arr[:, :, 0] > 150) & (arr[:, :, 0] > arr[:, :, 1] + 40) & (arr[:, :, 0] > arr[:, :, 2] + 40)
        if red_mask.any():
            arr = arr.copy()
            arr[red_mask] = 255
            image = Image.fromarray(arr)

    config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    try:
        tesseract_cmd = _get_tesseract_command()
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        tessdata_prefix = _get_tessdata_prefix()
        if tessdata_prefix is not None:
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix
        text = pytesseract.image_to_string(image, lang="eng", config=config).strip()
        if text:
            return text
    except Exception:
        pass

    return _simple_numeric_ocr(image)


def find_marker_label(image, marker_px, marker_py, search_width=280, search_height=160):
    w, h = image.size
    x0 = max(0, marker_px - search_width // 2)
    x1 = min(w, marker_px + search_width // 2)
    y1 = max(0, marker_py - 10)
    y0 = max(0, y1 - search_height)
    if x0 >= x1 or y0 >= y1:
        return None

    candidates = [
        (x0, y0, x1, y1),
        (max(0, marker_px - search_width), y0, min(w, marker_px + search_width), y1),
        (max(0, marker_px - search_width // 2), max(0, y0 - search_height // 3), min(w, marker_px + search_width), y1),
    ]
    for box in candidates:
        bx0, by0, bx1, by1 = box
        if bx0 >= bx1 or by0 >= by1:
            continue
        label_crop = image.crop(box)
        text = ocr_image_to_text(label_crop)
        value = parse_numeric_label(text)
        if value is not None:
            return value
    return None


def select_measurement_marker(red_points, target_voltage):
    if not red_points:
        return None
    ordered_points = sorted(red_points, key=lambda point: point[0])
    target_index = min(range(len(MEASUREMENT_LEVELS)), key=lambda index: abs(MEASUREMENT_LEVELS[index] - target_voltage))
    target_index = min(target_index, len(ordered_points) - 1)
    return ordered_points[target_index]


def find_closest_data_point(mapped_points, target_x):
    if not mapped_points:
        return None
    return min(mapped_points, key=lambda item: abs(item[0] - target_x))


def interpolate_y_at_x(mapped_points, target_x):
    """
    Return an interpolated (x,y) pair at target_x using linear interpolation
    between the two mapped points that bracket target_x. If outside range,
    return the nearest endpoint.
    """
    if not mapped_points:
        return None
    pts = sorted(mapped_points, key=lambda p: p[0])
    if target_x <= pts[0][0]:
        return (float(pts[0][0]), float(pts[0][1]))
    if target_x >= pts[-1][0]:
        return (float(pts[-1][0]), float(pts[-1][1]))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= target_x <= x1 or x1 <= target_x <= x0:
            if x1 == x0:
                return (float(x0), float(y0))
            t = (target_x - x0) / (x1 - x0)
            y = y0 + t * (y1 - y0)
            return (float(target_x), float(y))
    return (float(pts[0][0]), float(pts[0][1]))


def find_closest_bar(bar_points, target_voltage):
    if not bar_points:
        return None
    return min(bar_points, key=lambda item: abs(item[0] - target_voltage))


def extract_error_bars(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    mask = (b > 150) & (g > 130) & (r > 100) & ((b - r) > 20)
    comps = connected_components_bool(mask)
    h, w, _ = arr.shape
    bars = []
    for x0, y0, x1, y1, area in comps:
        bw = x1 - x0
        bh = y1 - y0
        if area < 1000:
            continue
        if bw < 40 or bh < 20:
            continue
        if bw > 0.95 * w or bh > 0.2 * h:
            continue
        bars.append((x0, y0, x1, y1, area))
    bars.sort(key=lambda item: (item[1], item[0]))
    return bars


def extract_error_entries(image):
    bars = extract_error_bars(image)
    entries = []
    w, h = image.size
    for x0, y0, x1, y1, area in bars:
        label_x0 = max(0, x0 - 40)
        label_y0 = max(0, y0 - 20)
        label_x1 = min(w, x1 + 260)
        label_y1 = min(h, y1 + 20)
        label_crop = image.crop((label_x0, label_y0, label_x1, label_y1))
        text = ocr_image_to_text(label_crop, psm=7)
        error_value = parse_numeric_label(text)
        if error_value is None:
            label_crop = ImageOps.autocontrast(label_crop.convert("L")).resize(
                (max(1, label_crop.width * 3), max(1, label_crop.height * 3)),
                Image.Resampling.LANCZOS,
            )
            text = ocr_image_to_text(label_crop, psm=6)
            error_value = parse_numeric_label(text)
        entries.append(
            {
                "bbox": (x0, y0, x1, y1),
                "center_y": (y0 + y1) / 2.0,
                "center_x": (x0 + x1) / 2.0,
                "text": text,
                "error": error_value,
            }
        )
    entries.sort(key=lambda item: item["center_y"])
    return entries


def select_error_entry(entries, target_voltage):
    if not entries:
        return None
    target_index = min(range(len(ERROR_LEVELS)), key=lambda index: abs(ERROR_LEVELS[index] - target_voltage))
    target_index = min(target_index, len(entries) - 1)
    return entries[target_index]


def parse_range(text):
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid numeric value: {text}")


def save_points_csv(points, path):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for x, y in points:
            writer.writerow([f"{x:.6f}", f"{y:.6f}"])


def excel_column_name(index):
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def build_cell_ref(column_index, row_index):
    return f"{excel_column_name(column_index)}{row_index}"


def measurement_series_for_channel(pdf_path, channel, first_chart_page):
    page_measure = first_chart_page + (channel - 1) * 2
    image = render_pdf_page(pdf_path, page_measure, zoom=RENDER_ZOOM)
    regions = detect_plot_regions(image)
    if len(regions) == 1:
        x, y, cw, ch, area = regions[0]
        gray = pil_to_gray(image)
        h, w = gray.shape
        if x == 0 and y == 0 and cw == w and ch == h:
            bbox = content_bbox(image)
            if bbox is not None:
                bx, by, bw, bh = bbox
                regions = [(bx, by, bw, bh, bw * bh)]
    chart_index = 0
    if len(regions) > 1:
        chart_index = min(len(regions) - 1, channel - 1)
    region = regions[chart_index]
    cropped, _ = crop_region(image, region)
    red_points = sorted(extract_red_markers(cropped), key=lambda point: point[0])
    series = {}
    for index, point in enumerate(red_points[: len(MEASUREMENT_LEVELS)]):
        label = find_marker_label(cropped, point[0], point[1])
        value = parse_numeric_label(str(label)) if label is not None else None
        if value is not None:
            series[MEASUREMENT_LEVELS[index]] = value
    return series


def error_series_for_channel(pdf_path, channel, first_chart_page):
    page_error = first_chart_page + (channel - 1) * 2 + 1
    image = render_pdf_page(pdf_path, page_error, zoom=RENDER_ZOOM)
    regions = detect_plot_regions(image)
    if regions:
        bbox = content_bbox(image)
        if bbox is not None:
            bx, by, bw, bh = bbox
            regions = [(bx, by, bw, bh, bw * bh)]
    chart_index = 0
    if len(regions) > 1:
        chart_index = min(len(regions) - 1, channel - 1)
    region = regions[chart_index]
    cropped, _ = crop_region(image, region)
    entries = extract_error_entries(cropped)
    series = {}
    for index, entry in enumerate(entries[: len(ERROR_LEVELS)]):
        error_value = entry.get("error")
        if error_value is not None:
            series[ERROR_LEVELS[index]] = error_value
    return series


def build_excel_report(pdf_path, output_path, first_chart_page=2, channels=(1, 2, 3, 4)):
    measurement_data = {channel: measurement_series_for_channel(pdf_path, channel, first_chart_page) for channel in channels}
    error_data = {channel: error_series_for_channel(pdf_path, channel, first_chart_page) for channel in channels}

    planned_voltages = [0, 5, 10, 15, 20, 25, 30, 35]
    tolerance_text = "1.20%"
    fs_tolerance = 0.42

    match = re.search(r"VT(\d+)_(\d+)", os.path.basename(pdf_path))
    if match:
        sheet_title = f"VT {match.group(1)} {match.group(2)}"
    else:
        sheet_title = "VT 1004 1"
    title_text = f"measuring chain voltage {sheet_title}"

    headers = [
        ("planned voltage\n[V]", 1),
        ("tolerance\n[%]", 2),
        ("test system\nvalue\n[V]", 3),
        ("reference\nValue\n[V]", 4),
        ("i.O.", 5),
        ("n.i.\nO.", 6),
        ("test system\nvalue\n[V]", 7),
        ("reference\nValue\n[V]", 8),
        ("i.O.", 9),
        ("n.i.\nO.", 10),
        ("test system\nvalue\n[V]", 11),
        ("reference\nValue\n[V]", 12),
        ("i.O.", 13),
        ("n.i.\nO.", 14),
        ("test system\nvalue\n[V]", 15),
        ("reference\nValue\n[V]", 16),
        ("i.O.", 17),
        ("n.i.\nO.", 18),
    ]

    def make_row(cells):
        return "".join(cells)

    def cell(column, row, value, style=None, data_type=None):
        ref = build_cell_ref(column, row)
        style_attr = f' s="{style}"' if style is not None else ""
        if data_type == "inlineStr":
            return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_escape(str(value))}</t></is></c>'
        if value is None or value == "":
            return ""
        if data_type == "str":
            return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_escape(str(value))}</t></is></c>'
        return f'<c r="{ref}" t="n"{style_attr}><v>{value}</v></c>'

    rows_xml = []
    rows_xml.append(
        f'<row r="1" spans="1:18" ht="22" customHeight="1">'
        f'{cell(1, 1, title_text, style=1, data_type="inlineStr")}'
        f"</row>"
    )
    row2_cells = [
        cell(1, 2, "planned voltage\n[V]", style=2, data_type="inlineStr"),
        cell(2, 2, "tolerance\n[%]", style=2, data_type="inlineStr"),
        cell(3, 2, "Input 1", style=2, data_type="inlineStr"),
        cell(7, 2, "Input 2", style=2, data_type="inlineStr"),
        cell(11, 2, "Input 3", style=2, data_type="inlineStr"),
        cell(15, 2, "Input 4", style=2, data_type="inlineStr"),
    ]
    rows_xml.append(f'<row r="2" spans="1:18" ht="20" customHeight="1">{make_row(row2_cells)}</row>')

    row3_labels = {
        3: "test system\nvalue\n[V]",
        4: "reference\nValue\n[V]",
        5: "i.O.",
        6: "n.i.\nO.",
        7: "test system\nvalue\n[V]",
        8: "reference\nValue\n[V]",
        9: "i.O.",
        10: "n.i.\nO.",
        11: "test system\nvalue\n[V]",
        12: "reference\nValue\n[V]",
        13: "i.O.",
        14: "n.i.\nO.",
        15: "test system\nvalue\n[V]",
        16: "reference\nValue\n[V]",
        17: "i.O.",
        18: "n.i.\nO.",
    }
    row3_cells = [cell(col, 3, text, style=3, data_type="inlineStr") for col, text in row3_labels.items()]
    rows_xml.append(f'<row r="3" spans="1:18" ht="34" customHeight="1">{"".join(row3_cells)}</row>')

    for row_offset, planned_voltage in enumerate(planned_voltages, start=4):
        cells = [
            cell(1, row_offset, f"{planned_voltage} V", style=4, data_type="inlineStr"),
            cell(2, row_offset, tolerance_text, style=4, data_type="inlineStr"),
        ]
        for channel in channels:
            measured_value = measurement_data.get(channel, {}).get(planned_voltage)
            error_value = error_data.get(channel, {}).get(planned_voltage)
            if measured_value is None:
                cells.extend(
                    [
                        cell(3 + (channel - 1) * 4, row_offset, "", style=4, data_type="inlineStr"),
                        cell(4 + (channel - 1) * 4, row_offset, f"{planned_voltage} V", style=4, data_type="inlineStr"),
                        cell(5 + (channel - 1) * 4, row_offset, "", style=4, data_type="inlineStr"),
                        cell(6 + (channel - 1) * 4, row_offset, "", style=4, data_type="inlineStr"),
                    ]
                )
                continue

            within_tolerance = abs(measured_value - planned_voltage) <= fs_tolerance
            cells.extend(
                [
                    cell(3 + (channel - 1) * 4, row_offset, f"{measured_value:.5f}V", style=4, data_type="inlineStr"),
                    cell(4 + (channel - 1) * 4, row_offset, f"{planned_voltage} V", style=4, data_type="inlineStr"),
                    cell(5 + (channel - 1) * 4, row_offset, "X" if within_tolerance else "", style=5 if within_tolerance else 4, data_type="inlineStr"),
                    cell(6 + (channel - 1) * 4, row_offset, "" if within_tolerance else "X", style=6 if not within_tolerance else 4, data_type="inlineStr"),
                ]
            )
        rows_xml.append(f'<row r="{row_offset}" spans="1:18" ht="18" customHeight="1">{"".join(cells)}</row>')

    sheet_data = "".join(rows_xml)
    cols_xml = "".join(
        [
            '<col min="1" max="1" width="14" customWidth="1"/>',
            '<col min="2" max="2" width="12" customWidth="1"/>',
            '<col min="3" max="18" width="14" customWidth="1"/>',
        ]
    )
    merges = [
        "A1:R1",
        "C2:F2",
        "G2:J2",
        "K2:N2",
        "O2:R2",
    ]
    merge_xml = "".join(f'<mergeCell ref="{ref}"/>' for ref in merges)

    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:R11"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="3" topLeftCell="A4" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{cols_xml}</cols>
  <sheetData>{sheet_data}</sheetData>
  <mergeCells count="{len(merges)}">{merge_xml}</mergeCells>
</worksheet>
"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="14"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border>
      <left/><right/><top/><bottom/><diagonal/>
    </border>
    <border>
      <left style="thin"><color rgb="FF000000"/></left>
      <right style="thin"><color rgb="FF000000"/></right>
      <top style="thin"><color rgb="FF000000"/></top>
      <bottom style="thin"><color rgb="FF000000"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="7">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="1" xfId="0" applyFont="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="1" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center"/>
    </xf>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center"/>
    </xf>
  </cellXfs>
</styleSheet>
"""

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="VT1004 1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""

    workbook_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

    root_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application>
</Properties>
"""

    core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Measurement export</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>
"""

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def region_to_dict(region):
    x, y, cw, ch, area = region
    return {"x": int(x), "y": int(y), "width": int(cw), "height": int(ch), "area": int(area)}


def interactive_mode(pdf_path, first_chart_page, x_min, x_max, y_min, y_max, error_x_min, error_x_max, error_y_min, error_y_max):
    """Interactive mode: ask user for channel and voltage, return measurement and error."""
    print("\n=== PDF Calibration Data Extractor - Interactive Mode ===")
    print(f"PDF: {pdf_path}")
    print(f"Measurement axes: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"Error axes: x=[{error_x_min}, {error_x_max}], y=[{error_y_min}, {error_y_max}]")
    print("Type 'exit' or 'quit' to stop\n")
    
    doc = fitz.open(pdf_path)
    try:
        while True:
            try:
                ch_input = input("Enter channel number (1-4): ").strip()
                if ch_input.lower() in ('exit', 'quit'):
                    print("Exiting.")
                    break
                channel = int(ch_input)
                if channel not in (1, 2, 3, 4):
                    print("Channel must be 1, 2, 3, or 4")
                    continue
                
                v_input = input(f"Enter target voltage: ").strip()
                if v_input.lower() in ('exit', 'quit'):
                    print("Exiting.")
                    break
                target_voltage = float(v_input)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue
            
            # Process measurement
            page_measure = first_chart_page + (channel - 1) * 2
            page_obj = doc[page_measure]
            image = render_pdf_page(pdf_path, page_measure, zoom=RENDER_ZOOM)
            regions = detect_plot_regions(image)
            if len(regions) == 1:
                x, y, cw, ch, area = regions[0]
                gray = pil_to_gray(image)
                h, w = gray.shape
                if x == 0 and y == 0 and cw == w and ch == h:
                    bbox = content_bbox(image)
                    if bbox is not None:
                        bx, by, bw, bh = bbox
                        regions = [(bx, by, bw, bh, bw * bh)]
            
            chart_index = 0
            if len(regions) > 1:
                chart_index = min(len(regions) - 1, channel - 1)
            region = regions[chart_index]
            cropped, region_size = crop_region(image, region)
            raw_points = extract_line_points(cropped)
            mapped_points = map_points_to_data(raw_points, region_size, x_min, x_max, y_min, y_max)
            direct_label = None
            if raw_points and target_voltage is not None:
                red_points = extract_red_markers(cropped)
                if red_points and len(red_points) >= 3:
                    mapped_red = map_points_to_data(red_points, region_size, x_min, x_max, y_min, y_max)
                    if mapped_red:
                        selected_point = select_measurement_marker(red_points, target_voltage)
                        if selected_point is not None:
                            direct_label = find_marker_label(cropped, selected_point[0], selected_point[1])
            
            # Fallback full-width extraction if needed
            if mapped_points:
                xs = [p[0] for p in mapped_points]
                min_x = min(xs)
                max_x = max(xs)
                if target_voltage < min_x or target_voltage > max_x:
                    raw_points_full = extract_line_points_full(cropped)
                    if raw_points_full:
                        mapped_full = map_points_to_data(raw_points_full, region_size, x_min, x_max, y_min, y_max)
                        if mapped_full:
                            mapped_points = mapped_full
            
            if direct_label is not None:
                measurement = (float(target_voltage), float(direct_label))
            else:
                measurement = None
            
            # Process error
            page_error = first_chart_page + (channel - 1) * 2 + 1
            image_err = render_pdf_page(pdf_path, page_error, zoom=RENDER_ZOOM)
            regions_err = detect_plot_regions(image_err)
            if regions_err:
                bbox = content_bbox(image_err)
                if bbox is not None:
                    bx, by, bw, bh = bbox
                    regions_err = [(bx, by, bw, bh, bw * bh)]
            
            chart_index_err = 0
            if len(regions_err) > 1:
                chart_index_err = min(len(regions_err) - 1, channel - 1)
            region_err = regions_err[chart_index_err]
            cropped_err, region_size_err = crop_region(image_err, region_err)
            entries = extract_error_entries(cropped_err)
            error_val = select_error_entry(entries, target_voltage)
            
            # Display results
            print(f"\n--- Channel {channel} @ {target_voltage}V ---")
            if measurement:
                print(f"  Measurement: {measurement[1]:.6f}V")
            else:
                print(f"  Measurement: Not found")
            
            if error_val and error_val.get("error") is not None:
                print(f"  Error: {error_val['error']:.8f}")
            else:
                print(f"  Error: Not found")
            print()
            
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error processing request: {e}")


def main():
    parser = argparse.ArgumentParser(description="PDF graph calibration data extractor")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=None, help="Page index (0-based). If omitted, --channel and --first-chart-page are used")
    parser.add_argument("--channel", type=int, default=1, help="Channel number (1-based). Mapped to chart pages when --page is omitted")
    parser.add_argument("--first-chart-page", type=int, default=2, help="Page index (0-based) of channel 1 measurement chart. Channels increment by two pages")
    parser.add_argument("--chart-type", choices=["measurement", "error", "both"], default="measurement", help="Which chart type to extract: measurement, error, or both")
    parser.add_argument("--x-min", type=parse_range, default=0.0, help="Minimum x-axis value shown on the measurement chart")
    parser.add_argument("--x-max", type=parse_range, default=1.0, help="Maximum x-axis value shown on the measurement chart")
    parser.add_argument("--y-min", type=parse_range, default=0.0, help="Minimum y-axis value shown on the measurement chart")
    parser.add_argument("--y-max", type=parse_range, default=1.0, help="Maximum y-axis value shown on the measurement chart")
    parser.add_argument("--error-x-min", type=parse_range, default=-0.01, help="Minimum x-axis value shown on the error chart")
    parser.add_argument("--error-x-max", type=parse_range, default=0.01, help="Maximum x-axis value shown on the error chart")
    parser.add_argument("--error-y-min", type=parse_range, default=-35.0, help="Minimum y-axis value shown on the error chart")
    parser.add_argument("--error-y-max", type=parse_range, default=35.0, help="Maximum y-axis value shown on the error chart")
    parser.add_argument("--target-x", type=parse_range, help="Target voltage to query from the extracted graph")
    parser.add_argument("--all-channels", action="store_true", help="Extract the requested --target-x value for channels 1..4 using --first-chart-page mapping")
    parser.add_argument("--channels", help="Comma-separated channel numbers to extract (e.g. 1,2,3)")
    parser.add_argument("--output-csv", help="Write extracted points to CSV")
    parser.add_argument("--output-xlsx", help="Write extracted measurements to an Excel workbook")
    parser.add_argument("--debug-json", help="Write debug metadata to JSON")
    parser.add_argument("--list-channels", action="store_true", help="List detected plot regions on the page")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for channel and voltage inputs")
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(args.pdf, args.first_chart_page, args.x_min, args.x_max, args.y_min, args.y_max, 
                        args.error_x_min, args.error_x_max, args.error_y_min, args.error_y_max)
        return

    if args.output_xlsx:
        if args.channels:
            try:
                channels = [int(s) for s in args.channels.split(',') if s.strip()]
            except Exception:
                channels = [1, 2, 3, 4]
        else:
            channels = [1, 2, 3, 4]
        build_excel_report(args.pdf, args.output_xlsx, first_chart_page=args.first_chart_page, channels=channels)
        print(json.dumps({"pdf": args.pdf, "output_xlsx": args.output_xlsx, "channels": channels}, indent=2))
        return

    doc = fitz.open(args.pdf)

    def get_page_for(channel, chart_type):
        if args.page is not None:
            if chart_type == "measurement":
                return args.page
            if args.chart_type == "both":
                return args.page + 1
            return args.page
        base = max(0, args.first_chart_page + (channel - 1) * 2)
        if chart_type == "measurement":
            return base
        return base + 1

    def process_page(page_to_use, chart_type, channel):
        page_obj = doc[page_to_use]
        image = render_pdf_page(args.pdf, page_to_use, zoom=RENDER_ZOOM)
        regions = detect_plot_regions(image)
        if len(regions) == 1:
            x, y, cw, ch, area = regions[0]
            gray = pil_to_gray(image)
            h, w = gray.shape
            if x == 0 and y == 0 and cw == w and ch == h:
                bbox = content_bbox(image)
                if bbox is not None:
                    bx, by, bw, bh = bbox
                    regions = [(bx, by, bw, bh, bw * bh)]

        if chart_type == 'error' and regions:
            bbox = content_bbox(image)
            if bbox is not None:
                bx, by, bw, bh = bbox
                regions = [(bx, by, bw, bh, bw * bh)]

        chart_index = 0
        if len(regions) > 1:
            chart_index = min(len(regions) - 1, args.channel - 1)
        region = regions[chart_index]
        cropped, region_size = crop_region(image, region)
        if chart_type == 'error':
            entries = extract_error_entries(cropped)
            best = None
            if args.target_x is not None:
                best = select_error_entry(entries, args.target_x)
            query = None
            if best is not None and best.get("error") is not None:
                query = {
                    'target_voltage': float(args.target_x),
                    'voltage': float(ERROR_LEVELS[min(range(len(ERROR_LEVELS)), key=lambda index: abs(ERROR_LEVELS[index] - args.target_x))]),
                    'error': float(best["error"]),
                    'distance': 0.0
                }
            return {
                'channel': channel,
                'chart_type': chart_type,
                'page': page_to_use,
                'region': region_to_dict(region),
                'points_extracted': len(entries),
                'mapped_points': entries if args.debug_json else None,
                'query': query
            }
        raw_points = extract_line_points(cropped)
        mapped_points = map_points_to_data(raw_points, region_size, args.x_min, args.x_max, args.y_min, args.y_max)
        direct_label = None
        if chart_type == 'measurement' and args.target_x is not None:
            red_points = extract_red_markers(cropped)
            if red_points and len(red_points) >= 3:
                mapped_red = map_points_to_data(red_points, region_size, args.x_min, args.x_max, args.y_min, args.y_max)
                if mapped_red:
                    selected_point = select_measurement_marker(red_points, args.target_x)
                    if selected_point is not None:
                        direct_label = find_marker_label(cropped, selected_point[0], selected_point[1])
        # If the requested target x lies outside the extracted x-range, try a full-width extraction
        if args.target_x is not None and mapped_points:
            xs = [p[0] for p in mapped_points]
            min_x = min(xs)
            max_x = max(xs)
            if args.target_x < min_x or args.target_x > max_x:
                raw_points_full = extract_line_points_full(cropped)
                if raw_points_full:
                    mapped_full = map_points_to_data(raw_points_full, region_size, args.x_min, args.x_max, args.y_min, args.y_max)
                    if mapped_full:
                        mapped_points = mapped_full
        query = None
        if args.target_x is not None and direct_label is not None:
            query = {'x': float(args.target_x), 'y': float(direct_label), 'distance': 0.0}
        return {
            'channel': channel,
            'chart_type': chart_type,
            'page': page_to_use,
            'region': region_to_dict(region),
            'points_extracted': len(mapped_points),
            'mapped_points': mapped_points if (args.debug_json or args.output_csv) else None,
            'query': query
        }

    def process_channel(channel):
        result = {'channel': channel}
        if args.chart_type in ("measurement", "both"):
            page_measure = get_page_for(channel, "measurement")
            result['measurement'] = process_page(page_measure, "measurement", channel)
        if args.chart_type in ("error", "both"):
            page_error = get_page_for(channel, "error")
            result['error'] = process_page(page_error, "error", channel)
        return result

    channels_to_run = None
    if args.all_channels:
        channels_to_run = [1, 2, 3, 4]
    elif args.channels:
        try:
            channels_to_run = [int(s) for s in args.channels.split(',') if s.strip()]
        except Exception:
            channels_to_run = [args.channel]

    if channels_to_run:
        results = [process_channel(ch) for ch in channels_to_run]
        if args.debug_json:
            with open(args.debug_json, 'w', encoding='utf-8') as f:
                json.dump({'pdf': args.pdf, 'channels': results}, f, indent=2)
        print(json.dumps({'pdf': args.pdf, 'channels': results}, indent=2))
        return

    if args.list_channels:
        page_to_use = args.page if args.page is not None else get_page_for(args.channel, args.chart_type)
        image = render_pdf_page(args.pdf, page_to_use, zoom=RENDER_ZOOM)
        regions = detect_plot_regions(image)
        print(json.dumps({'pdf': args.pdf, 'page': page_to_use, 'regions': [region_to_dict(region) for region in regions]}, indent=2))
        return

    if args.chart_type == "both":
        measurement_page = args.page if args.page is not None else get_page_for(args.channel, "measurement")
        error_page = (args.page + 1) if args.page is not None else get_page_for(args.channel, "error")
        single = {
            'channel': args.channel,
            'measurement': process_page(measurement_page, "measurement", args.channel),
            'error': process_page(error_page, "error", args.channel),
        }
    else:
        page_to_use = args.page if args.page is not None else get_page_for(args.channel, args.chart_type)
        single = process_page(page_to_use, args.chart_type, args.channel)

    if args.output_csv and single.get('mapped_points') is not None:
        save_points_csv(single.get('mapped_points', []), args.output_csv)
    if args.debug_json:
        with open(args.debug_json, 'w', encoding='utf-8') as f:
            json.dump({'pdf': args.pdf, 'chart': single}, f, indent=2)
    print(json.dumps(single, indent=2))


if __name__ == "__main__":
    main()
