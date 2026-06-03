from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
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

VALID_PRINT_TYPES = [
    "Screen Printing",
    "Screen Print",
    "Embroidery",
    "DTF",
    "DTG",
    "Full Color",
    "Vinyl",
    "Sublimation",
    "Patch",
    "Digital Print",
]

PRODUCT_KEYWORDS = [
    "polo",
    "shirt",
    "tee",
    "hoodie",
    "crewneck",
    "sweatshirt",
    "quarter zip",
    "qz",
    "hat",
    "cap",
    "short",
    "pant",
    "tote",
    "bag",
    "banner",
    "sticker",
    "poster",
]

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

PLACEHOLDER_RE = re.compile(r"\[\[.*?\]\]")


def clean_text(value: Any) -> str:
    return str(value or "").replace("\u00a0", " ").strip()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def clean_pdf_text(text: str) -> str:
    text = text or ""
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("′", "'")
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\u00a0", " ")
    return normalize_space(text)


def is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    return bool(PLACEHOLDER_RE.search(str(value)))


def normalize_print_type(value: Optional[str]) -> Optional[str]:
    if not value or is_placeholder(value):
        return None

    key = normalize_space(value).lower()

    for raw, normalized in PRINT_TYPE_MAP.items():
        if raw in key:
            return normalized

    return "OTHER"


def extract_specialty_print(value: Optional[str]) -> List[str]:
    if not value or is_placeholder(value):
        return []

    key = normalize_space(value).lower()
    found = []

    for raw, normalized in SPECIALTY_KEYWORDS.items():
        if raw in key:
            found.append(normalized)

    return sorted(set(found))


def normalize_placement(value: Optional[str]) -> Optional[str]:
    if not value or is_placeholder(value):
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

    title = normalize_space(product_title).lower()

    if "polo" in title:
        return "POLO"
    if "hoodie" in title:
        return "HOODIE"
    if "crewneck" in title:
        return "CREWNECK"
    if "sweatshirt" in title:
        return "SWEATSHIRT"
    if "quarter zip" in title or "qz" in title:
        return "QUARTER_ZIP"
    if "shirt" in title or "tee" in title:
        return "SHIRT"
    if "hat" in title or "cap" in title:
        return "HAT"
    if "short" in title:
        return "SHORTS"
    if "pant" in title:
        return "PANTS"
    if "tote" in title:
        return "TOTE"
    if "bag" in title:
        return "BAG"
    if "banner" in title:
        return "BANNER"
    if "sticker" in title:
        return "STICKER"
    if "poster" in title:
        return "POSTER"

    return "OTHER"


def extract_colors(text: str) -> List[str]:
    text = clean_pdf_text(text)
    colors = []

    pantones = re.findall(r"(?:PANTONE\s*)?\b\d{3,4}\s*C\b", text, flags=re.I)

    for color in pantones:
        color = normalize_space(color).upper()

        if color in ["000", "000C", "PANTONE 000", "PANTONE 000C"]:
            continue

        if not color.startswith("PANTONE"):
            color = f"PANTONE {color}"

        colors.append(color)

    if re.search(r"\bSPOT\s+WHITE\b", text, flags=re.I):
        colors.append("SPOT WHITE")
    elif re.search(r"\bWHITE\b", text, flags=re.I):
        colors.append("WHITE")

    clean_colors = []

    for color in colors:
        color = normalize_space(color).upper()

        if color in ["000", "000C", "PANTONE 000", "PANTONE 000C"]:
            continue

        clean_colors.append(color)

    return sorted(set(clean_colors))


DIMENSION_RE = re.compile(
    r"""
    (?:Dimensions:\s*)?
    (?:Approx\.?\s*)?
    (?P<width>\d+(?:\.\d+)?)
    \s*["']?
    \s*w
    \s*x
    \s*
    (?P<height>\d+(?:\.\d+)?)
    \s*["']?
    \s*h
    (?:\s*-\s*(?P<placement>.*?))?
    (?=
        \s+Print\s*Type:
        |\s+Dimensions:
        |\s+White
        |\s+Black
        |\s+Pantone
        |\s+PANTONE
        |\s+Proof
        |$
    )
    """,
    re.I | re.X,
)


def extract_dimension_specs(text: str) -> List[Dict[str, Any]]:
    text = clean_pdf_text(text)
    results = []

    for match in DIMENSION_RE.finditer(text):
        width = float(match.group("width"))
        height = float(match.group("height"))
        placement = match.group("placement")

        if placement:
            placement = normalize_space(placement)

        if is_placeholder(placement):
            placement = None

        results.append(
            {
                "width": width,
                "height": height,
                "placement": placement,
            }
        )

    return results


def extract_print_types(text: str) -> List[str]:
    text = clean_pdf_text(text)
    results = []

    pattern = re.compile(
        r"Print\s*Type:\s*(?P<value>.*?)(?=\s+Print\s*Type:|\s+Dimensions:|\s+Size:|\s+Approx\.?:|\s+Proof|$)",
        flags=re.I,
    )

    for match in pattern.finditer(text):
        value = normalize_space(match.group("value"))

        if is_placeholder(value):
            continue

        matched_valid = None

        for pt in VALID_PRINT_TYPES:
            if re.search(rf"\b{re.escape(pt)}\b", value, flags=re.I):
                matched_valid = pt
                break

        if matched_valid:
            results.append(matched_valid)

    if results:
        return results

    for pt in VALID_PRINT_TYPES:
        matches = re.findall(rf"\b{re.escape(pt)}\b", text, flags=re.I)
        for _ in matches:
            results.append(pt)

    return results


def orientation(width: Optional[float], height: Optional[float]) -> Optional[str]:
    if not width or not height:
        return None
    if width > height:
        return "HORIZONTAL"
    if height > width:
        return "VERTICAL"
    return "SQUARE"


def extract_product_title(full_text: str) -> Optional[str]:
    lines = [
        normalize_space(x)
        for x in re.split(r"\n|\r", full_text)
        if normalize_space(x)
    ]

    if not lines:
        lines = [
            normalize_space(x)
            for x in re.split(r"(?=Print Type:|Dimensions:|Proof #)", full_text)
            if normalize_space(x)
        ]

    for line in lines:
        lower = line.lower()

        if lower.startswith(("print type", "dimensions", "size", "proof")):
            continue

        if any(keyword in lower for keyword in PRODUCT_KEYWORDS):
            return line

    text = clean_pdf_text(full_text)
    before_print_type = re.split(r"Print\s*Type:", text, flags=re.I)[0]
    candidates = [normalize_space(x) for x in before_print_type.split("  ") if normalize_space(x)]

    for candidate in candidates:
        lower = candidate.lower()
        if any(keyword in lower for keyword in PRODUCT_KEYWORDS):
            return candidate

    return None


def pdfplumber_text(pdf_path: str) -> str:
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            full_text.append(text)

    return "\n".join(full_text)


def pymupdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text("text") or "")

    doc.close()
    return "\n".join(text_parts)


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


def dedupe_decorations(decorations: List[Decoration]) -> List[Decoration]:
    seen = set()
    output = []

    for dec in decorations:
        key = (
            dec.raw_print_type,
            dec.print_type,
            dec.width_in,
            dec.height_in,
            dec.placement_key,
            tuple(dec.colors),
            dec.page,
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(dec)

    return output


def extract_decorations(pdf_path: str) -> ExtractResponse:
    plumber_text = pdfplumber_text(pdf_path)
    mupdf_text = pymupdf_text(pdf_path)

    full_text = plumber_text if len(plumber_text) >= len(mupdf_text) else mupdf_text
    full_text = clean_pdf_text(full_text)

    product_title = extract_product_title(full_text)
    product_type = extract_product_type(product_title)

    print_types = extract_print_types(full_text)
    dimensions = extract_dimension_specs(full_text)
    colors = extract_colors(full_text)

    decorations = []

    count = max(len(print_types), len(dimensions))

    for i in range(count):
        raw_print_type = print_types[i] if i < len(print_types) else None

        dim = dimensions[i] if i < len(dimensions) else {}

        width = dim.get("width")
        height = dim.get("height")
        placement_raw = dim.get("placement")

        if is_placeholder(raw_print_type):
            continue

        dec = Decoration(
            location=normalize_placement(placement_raw),
            raw_print_type=raw_print_type,
            print_type=normalize_print_type(raw_print_type),
            placement_raw=placement_raw,
            placement_key=normalize_placement(placement_raw),
            width_in=width,
            height_in=height,
            orientation=orientation(width, height),
            colors=colors,
            specialty_print=extract_specialty_print(raw_print_type),
            page=1,
        )

        dec.confidence = confidence_score(dec)

        if dec.raw_print_type or dec.width_in or dec.height_in:
            decorations.append(dec)

    decorations = dedupe_decorations(decorations)

    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    return ExtractResponse(
        filename=os.path.basename(pdf_path),
        pages=page_count,
        product_title=product_title,
        product_type=product_type,
        decorations=decorations,
        raw_text_preview=full_text[:1500],
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Fresh Prints PDF Spec Extractor",
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
