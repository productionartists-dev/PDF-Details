from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import fitz
import pdfplumber
import tempfile
import re
import os
import csv

app = FastAPI(title="Fresh Prints PDF Spec Extractor")


class Decoration(BaseModel):
    location: Optional[str] = None
    raw_print_type: Optional[str] = None
    print_type: Optional[str] = None

    placement_raw: Optional[str] = None
    placement_key: Optional[str] = None
    placement_offset_raw: Optional[str] = None

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

    product_code: Optional[str] = None
    product_title: Optional[str] = None
    catalog_product_name: Optional[str] = None
    product_type: Optional[str] = None
    product_classification_source: Optional[str] = None

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

PLACEHOLDER_RE = re.compile(r"\[\[.*?\]\]")
CATALOG_PATH = os.getenv("PRODUCT_CATALOG_PATH", "product_catalog.csv")


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
    text = text.replace("ﬁ", "fi")
    text = text.replace("\u00a0", " ")
    return normalize_space(text)


def normalize_code(value: Any) -> str:
    return normalize_space(value).upper()


def normalize_product_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    key = normalize_space(value).lower()

    if key in ["shirt", "t-shirt", "tee", "short sleeve", "long sleeve"]:
        return "SHIRT"
    if key == "polo":
        return "POLO"
    if key in ["hat", "cap"]:
        return "HAT"
    if key == "beanie":
        return "BEANIE"
    if key == "hoodie":
        return "HOODIE"
    if key in ["sweatshirt", "crewneck"]:
        return "SWEATSHIRT"
    if key in ["sweatpants/pants", "pants", "sweatpants", "joggers"]:
        return "SWEATPANTS_PANTS"
    if key == "shorts":
        return "SHORTS"
    if key in ["tank top", "tank"]:
        return "TANK_TOP"
    if key == "jersey":
        return "JERSEY"
    if key in ["tote bag", "tote"]:
        return "TOTE_BAG"
    if key in ["bag", "backpack", "duffel"]:
        return "BAG"
    if key in ["jacket/pullover", "jacket", "pullover", "quarter zip"]:
        return "JACKET_PULLOVER"
    if key == "banner":
        return "BANNER"
    if key == "sticker":
        return "STICKER"
    if key == "poster":
        return "POSTER"
    if key == "other":
        return "OTHER"

    return re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")


def load_product_catalog() -> Dict[str, Dict[str, str]]:
    catalog = {}

    if not os.path.exists(CATALOG_PATH):
        print(f"Product catalog not found at {CATALOG_PATH}.")
        return catalog

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            with open(
                CATALOG_PATH,
                "r",
                encoding=encoding,
                errors="ignore",
                newline="",
            ) as f:
                sample = f.read(4096)
                f.seek(0)

                delimiter = "\t" if "\t" in sample else ","
                reader = csv.DictReader(f, delimiter=delimiter)

                for row in reader:
                    product_code = normalize_code(
                        row.get("Product Code")
                        or row.get("product_code")
                        or row.get("Style Code")
                        or row.get("style_code")
                    )

                    if not product_code:
                        continue

                    product_name = normalize_space(
                        row.get("Product Name")
                        or row.get("product_name")
                        or row.get("Name")
                        or row.get("name")
                    )

                    classification = normalize_space(
                        row.get("Classification")
                        or row.get("classification")
                        or row.get("Product Type")
                        or row.get("product_type")
                        or "Other"
                    )

                    catalog[product_code] = {
                        "product_code": product_code,
                        "product_name": product_name,
                        "classification": classification,
                        "product_type": normalize_product_type(classification) or "OTHER",
                    }

                print(f"Loaded {len(catalog)} product catalog rows using {encoding}.")
                return catalog

        except Exception as e:
            print(f"Failed to load catalog using {encoding}: {e}")

    return {}


PRODUCT_CATALOG = load_product_catalog()


GENERIC_PRODUCT_CODE_RE = re.compile(
    r"(?<![A-Z0-9])("
    r"[A-Z]{1,6}\d{2,7}[A-Z0-9-]*"
    r"|\d{2,6}-\d{2,6}"
    r"|\d{3,7}"
    r")(?![A-Z0-9])",
    re.I,
)


def build_style_code_regex() -> re.Pattern:
    if PRODUCT_CATALOG:
        codes = sorted(PRODUCT_CATALOG.keys(), key=len, reverse=True)
        escaped = [re.escape(code) for code in codes if code]

        if escaped:
            return re.compile(
                r"(?<![A-Z0-9])(" + "|".join(escaped) + r")(?![A-Z0-9])",
                re.I,
            )

    return GENERIC_PRODUCT_CODE_RE


STYLE_CODE_RE = build_style_code_regex()


def find_product_code(text: str) -> Optional[str]:
    if not text:
        return None

    text_upper = clean_pdf_text(text).upper()

    catalog_matches = STYLE_CODE_RE.findall(text_upper)

    for match in catalog_matches:
        code = normalize_code(match)
        if code in PRODUCT_CATALOG:
            return code

    generic_matches = GENERIC_PRODUCT_CODE_RE.findall(text_upper)

    bad_codes = {
        "000",
        "000C",
        "001",
        "00",
    }

    for match in generic_matches:
        code = normalize_code(match)

        if code in bad_codes:
            continue

        if re.fullmatch(r"\d{6,}", code):
            continue

        if code.startswith("277") or code.startswith("368"):
            continue

        return code

    return None


def lookup_product_from_catalog(text: str) -> Dict[str, Optional[str]]:
    product_code = find_product_code(text)

    if product_code and product_code in PRODUCT_CATALOG:
        product = PRODUCT_CATALOG[product_code]

        return {
            "product_code": product_code,
            "catalog_product_name": product.get("product_name"),
            "product_type": product.get("product_type") or "OTHER",
            "source": "catalog",
        }

    return {
        "product_code": product_code,
        "catalog_product_name": None,
        "product_type": None,
        "source": None,
    }


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


def infer_print_location(
    product_type: Optional[str],
    width: Optional[float],
    height: Optional[float],
    placement_offset_raw: Optional[str],
    decoration_index: int,
    total_decorations: int,
    page_number: Optional[int] = None,
) -> Optional[str]:
    product_type = normalize_space(product_type or "").upper()
    offset = normalize_space(placement_offset_raw or "").lower()

    if page_number == 2 and total_decorations >= 2:
        return "FRONT"

    if page_number == 3 and total_decorations >= 2:
        return "BACK"

    if "left chest" in offset or "left pec" in offset:
        return "FRONT_LEFT_CHEST"
    if "right chest" in offset or "right pec" in offset:
        return "FRONT_RIGHT_CHEST"
    if "left sleeve" in offset:
        return "LEFT_SLEEVE"
    if "right sleeve" in offset:
        return "RIGHT_SLEEVE"
    if "front leg" in offset or "left leg" in offset:
        return "FRONT_LEG"
    if "back leg" in offset:
        return "BACK_LEG"
    if "front" in offset:
        return "FRONT"
    if "back" in offset:
        return "BACK"

    if product_type in ["HAT", "BEANIE"]:
        return "FRONT"

    if product_type in ["SHORTS", "SWEATPANTS_PANTS", "PANTS"]:
        if decoration_index == 0:
            return "FRONT_LEG"
        return "BACK_LEG"

    if product_type in [
        "SHIRT",
        "POLO",
        "HOODIE",
        "SWEATSHIRT",
        "JACKET_PULLOVER",
        "TANK_TOP",
        "JERSEY",
    ]:
        if width and height:
            if width <= 4.5 and height <= 4.5 and "collar" in offset:
                return "FRONT_LEFT_CHEST"

            if total_decorations > 1 and (width >= 8 or height >= 8):
                return "BACK"

            if total_decorations == 1 and (width >= 8 or height >= 8):
                return "FRONT"

        if "center" in offset:
            return "FRONT"

        if decoration_index == 0:
            return "FRONT"

        if decoration_index == 1:
            return "BACK"

    if product_type in ["TOTE_BAG", "BAG"]:
        if decoration_index == 0:
            return "FRONT"
        return "BACK"

    if "center" in offset:
        return "FRONT"

    if decoration_index == 0:
        return "FRONT"

    if decoration_index == 1:
        return "BACK"

    return "UNKNOWN"


def extract_product_type_from_text(product_title: Optional[str]) -> Optional[str]:
    if not product_title:
        return None

    title = normalize_space(product_title).lower()

    if re.search(r"\bpolo\b", title):
        return "POLO"
    if re.search(r"\bhoodie\b", title):
        return "HOODIE"
    if re.search(r"\bcrewneck\b", title):
        return "SWEATSHIRT"
    if re.search(r"\bsweatshirt\b", title):
        return "SWEATSHIRT"
    if re.search(r"\bquarter\s*zip\b|\bqz\b", title):
        return "JACKET_PULLOVER"
    if re.search(r"\bshirt\b|\btee\b|\bt-?shirt\b", title):
        return "SHIRT"
    if re.search(r"\bhat\b|\bcap\b", title):
        return "HAT"
    if re.search(r"\bshorts?\b", title):
        return "SHORTS"
    if re.search(r"\bpants?\b|\bsweatpants?\b|\bjoggers?\b", title):
        return "SWEATPANTS_PANTS"
    if re.search(r"\btote\b", title):
        return "TOTE_BAG"
    if re.search(r"\bbag\b|\bbackpack\b|\bduffel\b", title):
        return "BAG"
    if re.search(r"\bbanner\b", title):
        return "BANNER"
    if re.search(r"\bsticker\b", title):
        return "STICKER"
    if re.search(r"\bposter\b", title):
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
        \s+Dimensions:
        |\s+Print\s*Type:
        |\s+White
        |\s+Black
        |\s+Vegas
        |\s+Gold
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
            placement = re.sub(r"\s+Dimensions:.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+Print\s*Type:.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+White.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+Black.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+Vegas.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+Gold.*$", "", placement, flags=re.I).strip()
            placement = re.sub(r"\s+Proof.*$", "", placement, flags=re.I).strip()

        if is_placeholder(placement):
            placement = None

        results.append(
            {
                "width": width,
                "height": height,
                "placement_offset_raw": placement,
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
    original_text = full_text or ""
    catalog_lookup = lookup_product_from_catalog(original_text)
    product_code = catalog_lookup.get("product_code")
    catalog_name = catalog_lookup.get("catalog_product_name")

    lines = [
        normalize_space(x)
        for x in re.split(r"\n|\r", original_text)
        if normalize_space(x)
    ]

    if product_code:
        for line in lines:
            if product_code.upper() in line.upper():
                return line

    if catalog_name:
        return catalog_name

    for line in lines:
        lower = line.lower()

        if lower.startswith(("print type", "dimensions", "size", "proof")):
            continue

        if "bagels" in lower:
            continue

        if any(re.search(rf"\b{re.escape(keyword)}\b", lower) for keyword in PRODUCT_KEYWORDS):
            return line

    return None


def pdfplumber_pages_text(pdf_path: str) -> List[str]:
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages_text.append(text)

    return pages_text


def pymupdf_pages_text(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        pages_text.append(page.get_text("text") or "")

    doc.close()
    return pages_text


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
            dec.placement_offset_raw,
            tuple(dec.colors),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(dec)

    return output


def extract_decorations(pdf_path: str) -> ExtractResponse:
    plumber_pages = pdfplumber_pages_text(pdf_path)
    mupdf_pages = pymupdf_pages_text(pdf_path)

    page_count = max(len(plumber_pages), len(mupdf_pages))

    pages = []

    for i in range(page_count):
        plumber_text = plumber_pages[i] if i < len(plumber_pages) else ""
        mupdf_text = mupdf_pages[i] if i < len(mupdf_pages) else ""

        best_text = plumber_text if len(plumber_text) >= len(mupdf_text) else mupdf_text

        pages.append(best_text)

    full_text = "\n".join(pages)
    cleaned_text = clean_pdf_text(full_text)

    catalog_lookup = lookup_product_from_catalog(cleaned_text)

    product_code = catalog_lookup.get("product_code")
    catalog_product_name = catalog_lookup.get("catalog_product_name")

    product_title = extract_product_title(full_text)

    product_type = (
        catalog_lookup.get("product_type")
        or extract_product_type_from_text(product_title)
        or "OTHER"
    )

    product_classification_source = (
        catalog_lookup.get("source")
        if catalog_lookup.get("product_type")
        else "text_fallback"
    )

    colors = extract_colors(cleaned_text)

    decorations = []

    # Prefer detail pages, not summary page.
    # Page 1 usually contains both front/back together and can create duplicates.
    detail_page_indexes = list(range(1, page_count)) if page_count > 1 else [0]

    detail_page_decorations = []

    for page_index in detail_page_indexes:
        page_number = page_index + 1
        page_text = clean_pdf_text(pages[page_index])

        print_types = extract_print_types(page_text)
        dimensions = extract_dimension_specs(page_text)

        count = max(len(print_types), len(dimensions))

        for i in range(count):
            raw_print_type = print_types[i] if i < len(print_types) else None
            dim = dimensions[i] if i < len(dimensions) else {}

            width = dim.get("width")
            height = dim.get("height")
            placement_offset_raw = dim.get("placement_offset_raw")

            if is_placeholder(raw_print_type):
                continue

            location_key = infer_print_location(
                product_type=product_type,
                width=width,
                height=height,
                placement_offset_raw=placement_offset_raw,
                decoration_index=len(detail_page_decorations),
                total_decorations=max(1, page_count - 1),
                page_number=page_number,
            )

            if location_key in [None, "UNKNOWN"]:
                continue

            dec = Decoration(
                location=location_key,
                raw_print_type=raw_print_type,
                print_type=normalize_print_type(raw_print_type),
                placement_raw=location_key,
                placement_key=location_key,
                placement_offset_raw=placement_offset_raw,
                width_in=width,
                height_in=height,
                orientation=orientation(width, height),
                colors=colors,
                specialty_print=extract_specialty_print(raw_print_type),
                page=page_number,
            )

            dec.confidence = confidence_score(dec)

            if dec.raw_print_type or dec.width_in or dec.height_in:
                detail_page_decorations.append(dec)

    decorations = detail_page_decorations

    # Fallback to page 1 only if detail pages did not produce usable decorations.
    if not decorations:
        page_text = clean_pdf_text(pages[0]) if pages else cleaned_text
        print_types = extract_print_types(page_text)
        dimensions = extract_dimension_specs(page_text)

        count = max(len(print_types), len(dimensions))

        for i in range(count):
            raw_print_type = print_types[i] if i < len(print_types) else None
            dim = dimensions[i] if i < len(dimensions) else {}

            width = dim.get("width")
            height = dim.get("height")
            placement_offset_raw = dim.get("placement_offset_raw")

            if is_placeholder(raw_print_type):
                continue

            location_key = infer_print_location(
                product_type=product_type,
                width=width,
                height=height,
                placement_offset_raw=placement_offset_raw,
                decoration_index=i,
                total_decorations=count,
                page_number=1,
            )

            if location_key in [None, "UNKNOWN"]:
                continue

            dec = Decoration(
                location=location_key,
                raw_print_type=raw_print_type,
                print_type=normalize_print_type(raw_print_type),
                placement_raw=location_key,
                placement_key=location_key,
                placement_offset_raw=placement_offset_raw,
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
    real_page_count = doc.page_count
    doc.close()

    return ExtractResponse(
        filename=os.path.basename(pdf_path),
        pages=real_page_count,
        product_code=product_code,
        product_title=product_title,
        catalog_product_name=catalog_product_name,
        product_type=product_type,
        product_classification_source=product_classification_source,
        decorations=decorations,
        raw_text_preview=cleaned_text[:1500],
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Fresh Prints PDF Spec Extractor",
        "catalog_rows_loaded": len(PRODUCT_CATALOG),
        "catalog_path": CATALOG_PATH,
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
