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


# ==============================================================================
# Helper & Extraction Constants
# ==============================================================================

RENDER_ZOOM = 5.0
MEASUREMENT_LEVELS = [-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
PLANNED_VOLTAGES = [0, 5, 10, 15, 20, 25, 30, 35]
_OCR_TEMPLATE_CACHE = None


# ==============================================================================
# Outlier & Numeric Sanitization Helpers
# ==============================================================================

def parse_numeric_label(text):
    if not text:
        return None
    text = str(text).replace(",", ".")
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def correct_decimal_outlier(val, expected_val):
    if val is None:
        return None
    
    if abs(expected_val) < 1e-3:
        temp = abs(val)
        while temp >= 100.0:
            temp /= 10.0
            val /= 10.0
        return val

    ratio = abs(val / expected_val) if expected_val != 0 else abs(val)
    if ratio > 5.0:
        while abs(val / expected_val) > 2.0 and abs(val) > 1.0:
            val /= 10.0

    return val


# ==============================================================================
# PDF & Image Processing Routines
# ==============================================================================

def render_pdf_page(pdf_path, page_index=0, zoom=3.0):
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"PDF page index {page_index} out of range (0..{len(doc)-1})")
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


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
    return np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)


def block_mask(mask, block_size=(8, 8)):
    bh, bw = block_size
    padded = pad_to_multiple(mask, block_size)
    h, w = padded.shape
    blocks = padded.reshape(h // bh, bh, w // bw, bw)
    return np.any(blocks, axis=(1, 3))


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
        if region_area < min_area or cw < 0.15 * w or ch < 0.12 * h:
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
        overlap = any(x < ex + ecw and ex < x + cw and y < ey + ech and ey < y + ch for ex, ey, ecw, ech, _ in selected)
        if not overlap:
            selected.append(region)
        if len(selected) >= max_regions:
            break
    return selected if selected else [(0, 0, w, h, w * h)]


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


def extract_red_markers(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    red_score = r - np.maximum(g, b)
    mask = (r > 120) & (red_score > 50) & (g < 150) & (b < 150)
    if not np.any(mask):
        return []
    comps = connected_components_bool(mask)
    points = []
    for x0, y0, x1, y1, area in comps:
        cw = x1 - x0
        ch = y1 - y0
        if area < 15 or area > 5000:
            continue
        ratio = float(cw) / max(1, ch)
        if ratio < 0.4 or ratio > 2.5:
            continue
        marker_y, marker_x = np.where(mask[y0:y1, x0:x1])
        cx = x0 + int(np.mean(marker_x))
        cy = y0 + int(np.mean(marker_y))
        points.append((cx, cy))
    return points


# ==============================================================================
# OCR Engine & Label Reading
# ==============================================================================

def _get_tesseract_command():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def _get_tessdata_prefix():
    candidates = [
        os.path.join(os.path.dirname(__file__), "tessdata"),
        os.path.join(os.path.dirname(__file__), "entryPoint", "tessdata"),
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    return next((c for c in candidates if os.path.exists(os.path.join(c, "eng.traineddata"))), None)


def _build_ocr_templates():
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    font_path = next((path for path in font_candidates if os.path.exists(path)), None)
    if font_path is None:
        return {}

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
                variants.append(binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
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
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
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
    for character, template_list in templates.items():
        for template in template_list:
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

    return best_character if best_score >= 0.2 else None


def _simple_numeric_ocr(image):
    gray = np.asarray(image.convert("L") if image.mode != "L" else image).copy()
    if gray.size == 0:
        return None

    if float(gray.mean()) < 128:
        gray = 255 - gray

    mask = gray < 220
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

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

    return "".join(characters).replace("vv", "V") if characters else None


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


# ==============================================================================
# Simplified Fast Page Routing (Measurement Only)
# ==============================================================================

def normalized_channel_numbers(channels):
    cleaned = []
    for channel in channels or ():
        try:
            channel = int(channel)
        except (TypeError, ValueError):
            continue
        if channel > 0 and channel not in cleaned:
            cleaned.append(channel)
    return cleaned


def resolve_chart_region(image, channel=1):
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
    chart_index = min(len(regions) - 1, max(0, channel - 1)) if len(regions) > 1 else 0
    return regions[chart_index]


def classify_chart_image(image):
    region = resolve_chart_region(image)
    cropped, _ = crop_region(image, region)
    red_count = len(extract_red_markers(cropped))
    return "measurement" if red_count > 0 else None


def discover_chart_pages(pdf_path, first_chart_page=2):
    doc = fitz.open(pdf_path)
    measurement_pages = []
    for page_index in range(max(0, first_chart_page), len(doc)):
        image = render_pdf_page(pdf_path, page_index, zoom=RENDER_ZOOM)
        if classify_chart_image(image) == "measurement":
            measurement_pages.append(page_index)
    return measurement_pages


def page_for_channel(channel, first_chart_page, measurement_pages=None):
    if measurement_pages and 0 < channel <= len(measurement_pages):
        return measurement_pages[channel - 1]
    return max(0, first_chart_page + (channel - 1))


def group_points_by_x(points, tolerance=18):
    groups = []
    for point in sorted(points, key=lambda item: item[0]):
        if groups and abs(point[0] - groups[-1]["mean_x"]) <= tolerance:
            groups[-1]["points"].append(point)
            groups[-1]["mean_x"] = sum(p[0] for p in groups[-1]["points"]) / len(groups[-1]["points"])
        else:
            groups.append({"mean_x": float(point[0]), "points": [point]})
    return groups


def best_label_for_voltage(image, points, voltage):
    candidates = []
    for point in sorted(points, key=lambda item: item[1]):
        label = find_marker_label(image, point[0], point[1])
        value = parse_numeric_label(str(label)) if label is not None else None
        if value is not None:
            candidates.append(value)
    return min(candidates, key=lambda value: abs(value - voltage)) if candidates else None


# ==============================================================================
# Measurement Extraction Loop
# ==============================================================================

def measurement_series_for_channel(pdf_path, channel, first_chart_page, measurement_pages=None):
    page_measure = page_for_channel(channel, first_chart_page, measurement_pages)
    image = render_pdf_page(pdf_path, page_measure, zoom=RENDER_ZOOM)
    region = resolve_chart_region(image, channel)
    cropped, _ = crop_region(image, region)
    
    red_pts = extract_red_markers(cropped)
    
    if red_pts:
        xs = [p[0] for p in red_pts]
        ys = [p[1] for p in red_pts]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        is_vertical_stack = y_span > (x_span * 2)
    else:
        is_vertical_stack = False

    series = {}

    if is_vertical_stack:
        sorted_dots = sorted(red_pts, key=lambda p: p[1])
        for index, dot in enumerate(sorted_dots[: len(MEASUREMENT_LEVELS)]):
            voltage = MEASUREMENT_LEVELS[index]
            label = find_marker_label(cropped, dot[0], dot[1])
            value = parse_numeric_label(str(label)) if label is not None else None
            if value is not None:
                value = correct_decimal_outlier(value, voltage)
                series[voltage] = value
    else:
        red_groups = group_points_by_x(red_pts)
        for index, group in enumerate(red_groups[: len(MEASUREMENT_LEVELS)]):
            voltage = MEASUREMENT_LEVELS[index]
            value = best_label_for_voltage(cropped, group["points"], voltage)
            if value is not None:
                value = correct_decimal_outlier(value, voltage)
                series[voltage] = value

    return series


# ==============================================================================
# Excel Report Generation ("Channel" Output Formatting)
# ==============================================================================

def _build_styles_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
        '  <fonts count="4">\n'
        '    <font><sz val="11"/><name val="Calibri"/></font>\n'
        '    <font><b/><sz val="11"/><name val="Calibri"/></font>\n'
        '    <font><b/><sz val="11"/><color rgb="FF006100"/><name val="Calibri"/></font>\n'
        '    <font><b/><sz val="11"/><color rgb="FF9C0006"/><name val="Calibri"/></font>\n'
        '  </fonts>\n'
        '  <fills count="4">\n'
        '    <fill><patternFill patternType="none"/></fill>\n'
        '    <fill><patternFill patternType="gray125"/></fill>\n'
        '    <fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>\n'
        '    <fill><patternFill patternType="solid"><fgColor rgb="FFFFC7CE"/><bgColor indexed="64"/></patternFill></fill>\n'
        '  </fills>\n'
        '  <borders count="2">\n'
        '    <border><left/><right/><top/><bottom/></border>\n'
        '    <border>\n'
        '      <left style="thin"><color auto="1"/></left>\n'
        '      <right style="thin"><color auto="1"/></right>\n'
        '      <top style="thin"><color auto="1"/></top>\n'
        '      <bottom style="thin"><color auto="1"/></bottom>\n'
        '    </border>\n'
        '  </borders>\n'
        '  <cellXfs count="7">\n'
        '    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>\n'
        '    <xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'
        '    <xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>\n'
        '    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'
        '    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>\n'
        '    <xf numFmtId="0" fontId="2" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'
        '    <xf numFmtId="0" fontId="3" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'
        '  </cellXfs>\n'
        '</styleSheet>'
    )


def build_cell_ref(column_index, row_index):
    name = ""
    while column_index > 0:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return f"{name}{row_index}"


def build_excel_report(pdf_path, output_path, first_chart_page=2, channels=(1, 2, 3, 4)):
    measurement_pages = discover_chart_pages(pdf_path, first_chart_page)
    if channels:
        channels = normalized_channel_numbers(channels)
    else:
        channels = list(range(1, len(measurement_pages) + 1)) if measurement_pages else [1, 2, 3, 4]

    measurement_data = {
        channel: measurement_series_for_channel(pdf_path, channel, first_chart_page, measurement_pages)
        for channel in channels
    }
    
    planned_voltages = PLANNED_VOLTAGES
    tolerance_text = "1.20%"
    fs_tolerance = 0.42

    match = re.search(r"VT(\d+)_(\d+)", os.path.basename(pdf_path))
    sheet_title = f"VT {match.group(1)} {match.group(2)}" if match else "VT Report"
    title_text = f"measuring chain voltage {sheet_title}"

    total_columns = 2 + len(channels) * 4

    def cell(column, row, value, style=None, data_type=None):
        ref = build_cell_ref(column, row)
        style_attr = f' s="{style}"' if style is not None else ""
        if data_type == "inlineStr":
            return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_escape(str(value))}</t></is></c>'
        if value is None or value == "":
            return ""
        return f'<c r="{ref}" t="n"{style_attr}><v>{value}</v></c>'

    rows_xml = []
    rows_xml.append(
        f'<row r="1" spans="1:{total_columns}" ht="22" customHeight="1">'
        f'{cell(1, 1, title_text, style=1, data_type="inlineStr")}'
        f"</row>"
    )
    row2_cells = [
        cell(1, 2, "planned voltage\n[V]", style=2, data_type="inlineStr"),
        cell(2, 2, "tolerance\n[%]", style=2, data_type="inlineStr"),
    ]
    for index, channel in enumerate(channels):
        start_col = 3 + index * 4
       
        row2_cells.append(cell(start_col, 2, f"Channel {channel}", style=2, data_type="inlineStr"))
    rows_xml.append(f'<row r="2" spans="1:{total_columns}">' + "".join(row2_cells) + "</row>")

    row3_cells = ["", ""]
    for index, channel in enumerate(channels):
        row3_cells.extend([
            cell(3 + index * 4, 3, "test system\nvalue\n[V]", style=2, data_type="inlineStr"),
            cell(4 + index * 4, 3, "reference\nValue\n[V]", style=2, data_type="inlineStr"),
            cell(5 + index * 4, 3, "i.O.", style=2, data_type="inlineStr"),
            cell(6 + index * 4, 3, "n.i.\nO.", style=2, data_type="inlineStr"),
        ])
    rows_xml.append(f'<row r="3" spans="1:{total_columns}">' + "".join(row3_cells) + "</row>")

    for r_idx, v in enumerate(planned_voltages, start=4):
        row_cells = [
            cell(1, r_idx, f"{v} V", style=3, data_type="inlineStr"),
            cell(2, r_idx, tolerance_text, style=3, data_type="inlineStr"),
        ]
        for c_idx, ch in enumerate(channels):
            measured_val = measurement_data.get(ch, {}).get(v)
            val_str = f"{measured_val:.5f}V" if measured_val is not None else ""
            
            is_ok = True
            if measured_val is not None:
                diff = abs(measured_val - v)
                if diff > fs_tolerance:
                    is_ok = False
            
            io_mark = "X" if (measured_val is not None and is_ok) else ""
            nio_mark = "X" if (measured_val is not None and not is_ok) else ""

            row_cells.extend([
                cell(3 + c_idx * 4, r_idx, val_str, style=4, data_type="inlineStr"),
                cell(4 + c_idx * 4, r_idx, f"{v} V", style=3, data_type="inlineStr"),
                cell(5 + c_idx * 4, r_idx, io_mark, style=5 if io_mark else 3, data_type="inlineStr"),
                cell(6 + c_idx * 4, r_idx, nio_mark, style=6 if nio_mark else 3, data_type="inlineStr"),
            ])
        rows_xml.append(f'<row r="{r_idx}" spans="1:{total_columns}">' + "".join(row_cells) + "</row>")

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
        '  <sheetData>\n'
        + "".join(rows_xml) +
        '  </sheetData>\n'
        '</worksheet>'
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
        '  <sheets>\n'
        '    <sheet name="Report" sheetId="1" r:id="rId1"/>\n'
        '  </sheets>\n'
        '</workbook>'
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\n'
        '</Relationships>'
    )

    wb_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>\n'
        '  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\n'
        '</Relationships>'
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\n'
        '  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n'
        '  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>\n'
        '</Types>'
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml)
        zf.writestr("xl/styles.xml", _build_styles_xml())
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract measurement vector data from PDF reports.")
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Path to output Excel file", default="report.xlsx")
    args = parser.parse_args()
    build_excel_report(args.pdf, args.output)