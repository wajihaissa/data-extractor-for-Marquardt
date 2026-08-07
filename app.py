import json
import os
import re
import subprocess
import sys
import time
import traceback

import webview

# Matches "VT1007_1" style filenames -> reference "1007", number "1"
REFERENCE_PATTERN = re.compile(r"VT(\d+)_(\d+)", re.IGNORECASE)


def parse_reference(filename):
    """Pull the reference/number pair out of a report filename, e.g.
    'VT1007_1.pdf' -> ('1007', '1'). Returns (None, None) if it doesn't match.
    """
    match = REFERENCE_PATTERN.search(filename)
    if not match:
        return None, None
    return match.group(1), match.group(2)


# Make sure extractor.py (sitting next to this file, or bundled by PyInstaller) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extractor  # noqa: E402


def resource_path(relative_path):
    """Resolve a path both when running from source and when frozen by PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def app_data_dir():
    """Writable folder for generated reports + history — lives next to the exe,
    never inside the read-only PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "marquardt_reports")
    os.makedirs(path, exist_ok=True)
    return path


HISTORY_FILE = os.path.join(app_data_dir(), "history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entries):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def open_path(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class Api:
    """Exposed to the page as window.pywebview.api.<method>(...)"""

    def __init__(self):
        self._window = None

    def set_window(self, window):
        self._window = window

    # ---------- Extract Data ----------

    def select_pdf_file(self):
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("PDF files (*.pdf)", "All files (*.*)"),
        )
        if not result:
            return None
        return result[0]

    def extract_from_pdf(self, pdf_path, channels=None, first_chart_page=2):
        if not pdf_path or not os.path.exists(pdf_path):
            return {"status": "error", "message": "PDF not found on disk."}

        channels = tuple(channels) if channels else (1, 2, 3, 4)
        report_dir = app_data_dir()
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file_stamp = time.strftime("%Y%m%d-%H%M%S")
        output_name = f"{base_name}_{file_stamp}.xlsx"
        output_path = os.path.join(report_dir, output_name)

        try:
            extractor.build_excel_report(
                pdf_path,
                output_path,
                first_chart_page=first_chart_page,
                channels=channels,
            )
        except getattr(extractor, "FaultyReportError", ()) as exc:
            # Catch faulty report exception (e.g. multiple red dots/stacked readings)
            return {"status": "faulty_report", "message": str(exc)}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

        reference, number = parse_reference(base_name)
        entry = {
            "source_pdf": pdf_path,
            "source_name": os.path.basename(pdf_path),
            "output_path": output_path,
            "output_name": output_name,
            "channels": list(channels),
            "timestamp": timestamp,
            "reference": reference,
            "number": number,
        }
        history = load_history()
        history.insert(0, entry)
        save_history(history)

        return {"status": "ok", "report": entry}

    # ---------- Check Reports ----------

    def list_reports(self):
        history = load_history()
        changed = False
        for entry in history:
            if "reference" not in entry or "number" not in entry:
                reference, number = parse_reference(entry.get("source_name", ""))
                entry["reference"] = reference
                entry["number"] = number
                changed = True
        if changed:
            save_history(history)
        return history

    def search_reports(self, query="", reference="", number=""):
        history = self.list_reports()
        query = (query or "").strip().lower()
        reference = (reference or "").strip()
        number = (number or "").strip()

        def matches(entry):
            if reference and (entry.get("reference") or "") != reference:
                return False
            if number and (entry.get("number") or "") != number:
                return False
            if query:
                haystack = " ".join([
                    entry.get("source_name", ""),
                    entry.get("reference") or "",
                    entry.get("number") or "",
                ]).lower()
                if query not in haystack:
                    return False
            return True

        return [entry for entry in history if matches(entry)]

    def open_report(self, output_path):
        if not output_path or not os.path.exists(output_path):
            return {"status": "error", "message": "File no longer exists."}
        try:
            open_path(output_path)
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def reveal_reports_folder(self):
        try:
            open_path(app_data_dir())
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    html_path = resource_path("assets/marquardt-landing.html")
    api = Api()

    window = webview.create_window(
        "Marquardt — Vector Data Extraction",
        html_path,
        js_api=api,
        width=1280,
        height=820,
        min_size=(1040, 680),
    )
    api.set_window(window)
    webview.start()