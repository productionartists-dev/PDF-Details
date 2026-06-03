from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber
import tempfile
import re
import os

app = FastAPI(title="Fresh Prints PDF Spec Extractor")


class Decoration(BaseModel):
    location: Optional[str] = None
    raw_print_type: Optional[str] = None
    print_type: Optional[str] = None
    placement_raw: Optional[str] = None
    placement_key: Optional[str] = None
    width_in: Optional[float] = None
    height_in: Optional[float] = None
    orientation: Optional[str] = None
    colors: List[str] = []
    specialty_print: List[str] = []
    page: Optional[int] = None
    confidence: float = 0.0


class ExtractResponse(BaseModel):
    filename: str
    pages: int
    product_title: Optional[str] = None
    product_type: Optional[str] = None
    decorations: List[Decoration]
    raw_text_preview: Optional[str] = None


PRINT_TYPE_MAP = {
    "screen printing": "SCREEN_PRINT",
    "screen print": "SCREEN_PRINT",
    "embroidery": "EMBROIDERY",
    "dtf": "DTF_TRANSFER",
    "dtg": "DTG",
    "full color": "FULL_COLOR",
    "vinyl": "VINYL",
    "sublimation": "SUBLIMATION",
    "patch": "PATCH",
    "digital print": "DIGITAL_PRINT",
}

PRODUCT_KEYWORDS = {
    "polo": "POLO",
    "shirt": "SHIRT",
    "tee": "T_SHIRT",
    "hoodie": "HOODIE",
    "crewneck": "CREWNECK",
    "sweatshirt": "SWEATSHIRT",
    "quarter zip": "QUARTER_ZIP",
    "qz": "QUARTER_ZIP",
    "hat": "HAT",
    "cap": "CAP",
    "short": "SHORTS",
    "pant": "PANTS",
    "tote": "TOTE",
    "bag": "BAG",
    "banner": "BANNER",
    "sticker": "STICKER",
    "poster": "POSTER",
}

SPECIALTY_KEYWORDS = {
    "puff": "PUFF",
    "metallic": "METALLIC",
    "foil": "FOIL",
    "reflective": "REFLECTIVE",
    "glow": "GLOW",
    "high density": "HIGH_DENSITY",
    "applique": "APPLIQUE",
    "tackle twill": "TACKLE_TWILL",
    "flock": "FLOCK",
}


def clean_text(value: Any) -> str:
    return str(value or "").replace("\u00a0", " ").strip()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value))


def normalize_print_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    key = normalize_space(value).lower()

    for raw, normalized in PRINT_TYPE_MAP.items():
        if raw in key:
            return normalized

    return "OTHER"


def extract_specialty_print(value: Optional[str]) -> List[str]:
    if not value:
        return []

    key = normalize_space(value).lower()
    found = []

    for raw, normalized in SPECIALTY_KEYWORDS.items():
        if raw in key:
            found.append(normalized)

    return sorted(set(found))


def normalize_placement(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    key = normalize_space(value).lower()

    if "center" in key:
        return "CENTERED"
    if "collar" in key:
        return "FROM_COLLAR"
    if "bottom seam" in key:
        return "FROM_BOTTOM_SEAM"
    if "from seam" in key or "seam" in key:
        return "FROM_SEAM"
    if "left chest" in key or "left pec" in key:
        return "LEFT_CHEST"
    if "right chest" in key or "right pec" in key:
        return "RIGHT_CHEST"
    if "front" in key:
        return "FRONT"
    if "back" in key:
        return "BACK"
    if "left sleeve" in key:
        return "LEFT_SLEEVE"
    if "right sleeve" in key:
        return "RIGHT_SLEEVE"
    if "pocket" in key:
        return "POCKET"

    return re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")


def extract_product_type(product_title: Optional[str]) -> Optional[str]:
    if not product_title:
        return None

    key = normalize_space(product_title).lower()

    for raw, normalized in PRODUCT_KEYWORDS.items():
        if raw in key:
            return normalized

    return "OTHER"


def extract_colors(text: str) -> List[str]:
    colors = []

    pantones = re.findall(r"(?:PANTONE\s*)?\b\d{3,4}\s*C\b", text, flags=re.I)
    for color in pantones:
        color = normalize_space(color).upper()
        if not color.startswith("PANTONE"):
            color = f"PANTONE {color}"
        colors.append(color)

    if re.search(r"\bSPOT\s+WHITE\b", text, flags=re.I):
        colors.append("SPOT WHITE")
    elif re.search(r"\bWHITE\b", text, flags=re.I):
        colors.append("WHITE")

    return sorted(set(colors))


def parse_dimensions(text: str):
    patterns = [
        r"Dimensions:\s*([\d.]+)\s*[\"']?\s*w\s*x\s*([\d.]+)\s*[\"']?\s*h\s*(?:-\s*([^\n\r]+))?",
        r"Size:\s*([\d.]+)\s*[\"']?\s*w\s*x\s*([\d.]+)\s*[\"']?\s*h\s*(?:-\s*([^\n\r]+))?",
        r"Approx\.?:?\s*([\d.]+)\s*[\"']?\s*w\s*x\s*([\d.]+)\s*[\"']?\s*h\s*(?:-\s*([^\n\r]+))?",
        r"([\d.]+)\s*[\"']?\s*w\s*x\s*([\d.]+)\s*[\"']?\s*h\s*(?:-\s*([^\n\r]+))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            width = float(match.group(1))
            height = float(match.group(2))
            placement_raw = normalize_space(match.group(3)) if len(match.groups()) >= 3 and match.group(3) else None

            return width, height, placement_raw

    return None, None, None


def orientation(width: Optional[float], height: Optional[float]) -> Optional[str]:
    if not width or not height:
        return None
    if width > height:
        return "HORIZONTAL"
    if height > width:
        return "VERTICAL"
    return "SQUARE"


def extract_product_title(full_text: str) -> Optional[str]:
    lines = [normalize_space(x) for x in full_text.splitlines() if normalize_space(x)]

    for line in lines:
        lower = line.lower()
        if any(k in lower for k in PRODUCT_KEYWORDS.keys()):
            if not lower.startswith(("print type", "dimensions", "size", "proof")):
                return line

    return None


def pymupdf_text_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    blocks = []

    for page_index, page in enumerate(doc):
        page_blocks = page.get_text("blocks")

        for block in page_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            text = clean_text(text)
            if not text:
                continue

            blocks.append({
                "page": page_index + 1,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text,
            })

    doc.close()
    return blocks


def pdfplumber_text(pdf_path: str) -> str:
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            full_text.append(text)

    return "\n".join(full_text)


def group_nearby_spec_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Finds blocks around Print Type / Dimensions / Size labels.
    This is layout-aware, so it is more reliable than plain full-text regex.
    """
    spec_blocks = []

    for block in blocks:
        text = block["text"]

        if re.search(r"\bPrint Type\b|\bDimensions\b|\bSize\b|\bApprox\b", text, flags=re.I):
            page = block["page"]

            nearby = [
                b for b in blocks
                if b["page"] == page
                and abs(b["y0"] - block["y0"]) < 180
                and abs(b["x0"] - block["x0"]) < 350
            ]

            combined = "\n".join(b["text"] for b in sorted(nearby, key=lambda b: (b["y0"], b["x0"])))

            spec_blocks.append({
                "page": page,
                "x0": block["x0"],
                "y0": block["y0"],
                "text": combined,
            })

    # Dedupe similar blocks
    seen = set()
    deduped = []

    for block in spec_blocks:
        key = (block["page"], round(block["x0"], -1), round(block["y0"], -1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)

    return deduped


def extract_print_type_from_block(text: str) -> Optional[str]:
    inline = re.search(r"Print Type:\s*([^\n\r]+)", text, flags=re.I)
    if inline:
        candidate = normalize_space(inline.group(1))
        if not re.match(r"^(Dimensions|Size|Approx)\b", candidate, flags=re.I):
            return candidate

    lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]

    for i, line in enumerate(lines):
        if re.match(r"^Print Type:?\s*$", line, flags=re.I):
            for candidate in lines[i + 1:i + 5]:
                if re.match(r"^(Dimensions|Size|Approx|Shape):?", candidate, flags=re.I):
                    continue
                return candidate

    for line in lines:
        if normalize_print_type(line) not in [None, "OTHER"]:
            return line

    return None


def confidence_score(decoration: Decoration) -> float:
    score = 0

    if decoration.print_type:
        score += 25
    if decoration.width_in and decoration.height_in:
        score += 25
    if decoration.placement_key:
        score += 20
    if decoration.colors:
        score += 15
    if decoration.page:
        score += 5
    if decoration.raw_print_type:
        score += 10

    return min(100, score) / 100


def extract_decorations(pdf_path: str) -> ExtractResponse:
    blocks = pymupdf_text_blocks(pdf_path)
    full_text = pdfplumber_text(pdf_path)

    product_title = extract_product_title(full_text)
    product_type = extract_product_type(product_title)

    spec_blocks = group_nearby_spec_blocks(blocks)

    decorations = []

    for block in spec_blocks:
        block_text = block["text"]

        raw_print_type = extract_print_type_from_block(block_text)
        print_type = normalize_print_type(raw_print_type)

        width, height, placement_raw = parse_dimensions(block_text)

        colors = extract_colors(block_text) or extract_colors(full_text)
        specialty_print = extract_specialty_print(raw_print_type)

        dec = Decoration(
            location=normalize_placement(placement_raw),
            raw_print_type=raw_print_type,
            print_type=print_type,
            placement_raw=placement_raw,
            placement_key=normalize_placement(placement_raw),
            width_in=width,
            height_in=height,
            orientation=orientation(width, height),
            colors=colors,
            specialty_print=specialty_print,
            page=block["page"],
        )

        dec.confidence = confidence_score(dec)

        if dec.raw_print_type or dec.width_in or dec.height_in:
            decorations.append(dec)

    # Fallback if layout blocks fail
    if not decorations:
        raw_print_type = extract_print_type_from_block(full_text)
        width, height, placement_raw = parse_dimensions(full_text)

        dec = Decoration(
            location=normalize_placement(placement_raw),
            raw_print_type=raw_print_type,
            print_type=normalize_print_type(raw_print_type),
            placement_raw=placement_raw,
            placement_key=normalize_placement(placement_raw),
            width_in=width,
            height_in=height,
            orientation=orientation(width, height),
            colors=extract_colors(full_text),
            specialty_print=extract_specialty_print(raw_print_type),
            page=1,
        )

        dec.confidence = confidence_score(dec)
        decorations.append(dec)

    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    return ExtractResponse(
        filename=os.path.basename(pdf_path),
        pages=page_count,
        product_title=product_title,
        product_type=product_type,
        decorations=decorations,
        raw_text_preview=full_text[:1000],
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Fresh Prints PDF Spec Extractor"
    }


@app.post("/extract-specs", response_model=ExtractResponse)
async def extract_specs(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return extract_decorations(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
