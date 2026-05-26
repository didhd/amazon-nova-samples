"""Helpers for the hybrid Nova + Claude name-to-face matching pipeline.

Stage 1 calls Amazon Nova 2 Lite via the InvokeModel API following the
Nova 2 Developer Guide for object detection. Stage 2 calls Claude
Sonnet 4.6 with adaptive thinking via the Converse API and reasons over
the spatial layout to map names to faces.

Both stages keep all bounding boxes on Nova's native 0-1000 normalized grid,
so no coordinate conversion happens between calls.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)


# Stage 1 model. Pass `nova_model_id` to `run_pipeline` if you want to
# point at a different Nova multimodal model in the future.
NOVA_MODEL_ID = "us.amazon.nova-2-lite-v1:0"
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

COORD_SCALE = 1000


# System prompt that anchors Nova on the layout-analysis task. The wording
# mirrors what worked in the production yearbook pipeline.
NOVA_SYSTEM_PROMPT = """You are an expert at analyzing yearbook page layouts. You perform multiple tasks in a single pass:
1. Detect every photograph and return a tight bounding box for each one
2. Classify each photo by type and category
3. Extract the printed name(s) and a tight bounding box for each name
4. Extract the page title and a short page summary

CRITICAL RULES:
- Only detect ACTUAL PHOTOGRAPHS containing human faces or upper-body portraits.
- DO NOT detect text-only regions, captions, page numbers, body paragraphs, decorative borders, or logos as photos.
- A group photo is ONE photograph (one bbox). A portrait grid contains many separate photographs (one bbox per cell).
- Each portrait in a grid gets its OWN tight bounding box around the photograph frame only — do not include the printed name area inside the photo bbox.
- Each printed personal name gets its OWN tight bounding box around the printed text line only — do not include the photo above or below it inside the name bbox.

PORTRAIT GRID vs GROUP PHOTO:
- Portrait grid: each person has their own background/framing with visible borders between photos -> each cell = separate bbox.
- Group photo: people share the same background/scene, no internal borders -> entire photo = one bbox.

CLASSIFICATION:
- type: "portrait" | "group_photo" | "candid"
- category: a single word from this vocabulary when applicable: portraits, sports, academics, clubs, homecoming, cheerleading, prom, graduation, faculty, music, theater, dance, student_life. Otherwise pick a single fitting word.

OUTPUT: Return valid JSON only. No markdown, no code fences, no commentary."""


# User instruction. Asking for both photos and names in a single response keeps
# the pipeline at one Nova call per page.
NOVA_INSTRUCTION = """Identify every photograph (portrait, group photo, candid) and every printed personal name on this yearbook page. Use the [x1, y1, x2, y2] bbox format with integer coordinates scaled 0-1000 to image width and height.

Where to look for names:
- Under each portrait in a portrait grid, the printed name is a single line of type directly below the headshot.
- Beside or under group/candid photos, the names usually appear inside an italic caption such as "During a solo performance, ballerine Mira Solberg dances ...". Pull every personal name out of these captions and emit one entry per person, with the bbox tight around just that person's name span (not the whole caption sentence).
- A single italic caption can contain multiple names ("Henley Brookes and Lior Avani present a scene." -> two entries). A roster line under a group photo can also contain a comma-separated list of names — emit one entry per name.
- Do NOT include articles, verbs, page headers, page numbers, body paragraphs, or non-name text in the names array.

Bounding box rules:
- Coordinates are normalized: x in [0, 1000] left-to-right, y in [0, 1000] top-to-bottom.
- Measure the actual top-left and bottom-right of each element from the pixels. Different photos in a grid often start at different y positions; measure each one independently and do not output a uniform synthetic grid.
- Photo bbox must hug the photo frame only — exclude any printed name beneath or above it.
- Name bbox must hug the printed name span only — exclude the photo above or below, and exclude surrounding caption words that are not part of the name.
- For a name bbox, y1 must sit at the top of the tallest letter (cap height of the first letter) and y2 must sit at the bottom of the lowest descender (g, j, p, q, y) on that line. Do NOT clip the descenders by stopping at the baseline, and do NOT pad above the cap height into empty whitespace.
- Re-check each name bbox against the actual rendered glyphs before returning. If the printed name sits inside a multi-line italic caption, the bbox must wrap only the line that the name appears on, not the line above or below.

You MUST answer in JSON only and follow the output schema below exactly. Do not write any preamble, explanation, or markdown fences. Every bbox value must be a JSON integer literal (no quotes around the array, no quotes around individual numbers).

Output Schema:
{
  "page_title": "string or null",
  "page_category": "string or null",
  "page_summary": "string or null",
  "photos": [
    {
      "bbox": [x1, y1, x2, y2],
      "type": "portrait | group_photo | candid",
      "category": "single word",
      "summary": "<= 12 words"
    }
  ],
  "names": [
    {
      "text": "name as printed",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}"""


CLAUDE_SPATIAL_PROMPT_TEMPLATE = """You are matching printed names to faces on a yearbook-style page.

All coordinates are on a 0-1000 normalized grid (x left-to-right, y top-to-bottom) for both axes.

PHOTOS:
{photos_json}

NAMES:
{names_json}

For each printed name, decide which photo it belongs to. Portrait grids place the name directly under the headshot. Group and candid photos usually share a single italic caption that lists one or more names; map every name in that caption to the same face.

Return one JSON object exactly in this shape:

{{
  "associations": [
    {{
      "name": "<as printed>",
      "face_idx": <index into PHOTOS>,
      "confidence": <float 0-1>,
      "match_type": "portrait" | "group",
      "reasoning": "<short spatial explanation>"
    }}
  ],
  "unmatched_names": ["<name>", ...],
  "unmatched_face_indices": [<int>, ...]
}}

Rules:
- One name maps to exactly one face. A group or candid photo can receive several names; each one is its own association entry sharing the same face_idx.
- Use match_type "portrait" only when the name is the caption directly below or above an individual headshot.
- If a name does not clearly belong to any face, list it under unmatched_names.
- If a face has no readable caption, list its index in unmatched_face_indices.
- Output JSON only. No commentary, no code fences.
"""


def make_bedrock_client(region: str = "us-west-2"):
    return boto3.client("bedrock-runtime", region_name=region)


def load_image_bytes(path: str | Path) -> bytes:
    """Read an image file and return JPEG-encoded bytes."""
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def _parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object embedded in a model response.

    Nova occasionally returns JSON wrapped in code fences or with a trailing
    comma. Handle both cases before falling back to the raw string.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        candidate = re.sub(r"\n?```\s*$", "", candidate)

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
    raw = match.group()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Nova occasionally emits a stray quote after a JSON list, e.g.
        # `[38, 771, 222, 951]",`. Strip those before retrying. Also drop
        # any trailing commas before closing braces or brackets.
        cleaned = re.sub(r"\](\s*)\"", r"]\1", raw)
        cleaned = re.sub(r"\}(\s*)\"", r"}\1", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(cleaned)


def _retry(call, *, attempts: int = 3):
    """Retry a callable on transient JSON parse failures from Nova."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return call()
        except (json.JSONDecodeError, ValueError) as exc:
            last = exc
    assert last is not None
    raise last


def extract_photos_and_names(
    bedrock_runtime,
    image_bytes: bytes,
    *,
    model_id: str = NOVA_MODEL_ID,
    max_new_tokens: int = 8000,
    temperature: float = 0.0,
    attempts: int = 3,
) -> dict[str, Any]:
    """Stage 1 -- Nova InvokeModel returns photos, names, and page metadata.

    Uses the InvokeModel API (not Converse) because the Nova 2 Developer
    Guide recommends it for object-detection-style structured output. The
    request body follows messages-v1 schema with a separate system block.
    Retries up to `attempts` times if Nova returns invalid JSON.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    request_body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": NOVA_SYSTEM_PROMPT}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image_b64}}},
                    {"text": NOVA_INSTRUCTION},
                ],
            },
            # Prefill the assistant turn so Nova continues the JSON object
            # immediately, with no preamble. This is the technique recommended
            # in the Nova Developer Guide for structured-output prompts.
            {
                "role": "assistant",
                "content": [{"text": "{"}],
            },
        ],
        "inferenceConfig": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    def _call() -> dict[str, Any]:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        model_response = json.loads(response["body"].read())
        text = model_response["output"]["message"]["content"][0]["text"]
        # Re-attach the prefilled "{" since the assistant turn started there.
        if not text.lstrip().startswith("{"):
            text = "{" + text
        parsed = _parse_json_object(text)
        parsed.setdefault("page_title", None)
        parsed.setdefault("page_category", None)
        parsed.setdefault("page_summary", None)
        parsed.setdefault("photos", [])
        parsed.setdefault("names", [])
        parsed["_usage"] = model_response.get("usage", {})
        return parsed

    return _retry(_call, attempts=attempts)


def match_names_to_faces(
    bedrock_runtime,
    image_bytes: bytes,
    photos: list[dict[str, Any]],
    names: list[dict[str, Any]],
    *,
    model_id: str = CLAUDE_MODEL_ID,
    max_tokens: int = 64000,
    effort: str = "high",
) -> dict[str, Any]:
    """Stage 2 -- Claude Sonnet 4.6 with adaptive thinking maps names to faces."""
    if not photos:
        return {
            "associations": [],
            "unmatched_names": [n.get("text", "") for n in names],
            "unmatched_face_indices": [],
            "thinking_text": None,
            "_usage": {},
        }

    photos_for_prompt = [
        {
            "index": i,
            "bbox": p.get("bbox", [0, 0, 0, 0]),
            "type": p.get("type", "portrait"),
        }
        for i, p in enumerate(photos)
    ]
    names_for_prompt = [
        {"text": n.get("text", ""), "bbox": n.get("bbox", [0, 0, 0, 0])}
        for n in names
        if n.get("text")
    ]

    prompt = CLAUDE_SPATIAL_PROMPT_TEMPLATE.format(
        photos_json=json.dumps(photos_for_prompt, indent=2),
        names_json=json.dumps(names_for_prompt, indent=2),
    )

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens},
        additionalModelRequestFields={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        },
    )

    text = ""
    thinking_text = None
    for block in response["output"]["message"]["content"]:
        if "reasoningContent" in block:
            thinking_text = (
                block["reasoningContent"].get("reasoningText", {}).get("text")
            )
        elif "text" in block:
            text += block["text"]

    if not text.strip():
        return {
            "associations": [],
            "unmatched_names": [n["text"] for n in names_for_prompt],
            "unmatched_face_indices": list(range(len(photos))),
            "thinking_text": thinking_text,
            "_usage": response.get("usage", {}),
        }

    parsed = _parse_json_object(text)
    parsed.setdefault("associations", [])
    parsed.setdefault("unmatched_names", [])
    parsed.setdefault("unmatched_face_indices", [])
    parsed["thinking_text"] = thinking_text
    parsed["_usage"] = response.get("usage", {})
    return parsed


def run_pipeline(
    bedrock_runtime,
    image_path: str | Path,
    *,
    nova_model_id: str = NOVA_MODEL_ID,
    claude_model_id: str = CLAUDE_MODEL_ID,
) -> dict[str, Any]:
    """Run Stage 1 then Stage 2 and return one combined result."""
    image_bytes = load_image_bytes(image_path)
    extraction = extract_photos_and_names(
        bedrock_runtime, image_bytes, model_id=nova_model_id
    )
    matching = match_names_to_faces(
        bedrock_runtime,
        image_bytes,
        extraction["photos"],
        extraction["names"],
        model_id=claude_model_id,
    )
    return {
        "image_path": str(image_path),
        "page_title": extraction.get("page_title"),
        "page_category": extraction.get("page_category"),
        "page_summary": extraction.get("page_summary"),
        "photos": extraction["photos"],
        "names": extraction["names"],
        "associations": matching["associations"],
        "unmatched_names": matching["unmatched_names"],
        "unmatched_face_indices": matching["unmatched_face_indices"],
        "thinking_text": matching["thinking_text"],
        "usage": {
            "nova": extraction.get("_usage", {}),
            "claude": matching.get("_usage", {}),
        },
    }


def _denormalize(
    bbox_1000: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_1000
    return (
        int(round(x1 * width / COORD_SCALE)),
        int(round(y1 * height / COORD_SCALE)),
        int(round(x2 * width / COORD_SCALE)),
        int(round(y2 * height / COORD_SCALE)),
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def visualize_result(
    image_path: str | Path,
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Draw photo boxes and name->face links onto the original image."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font(max(14, height // 60))

    palette = [
        (220, 38, 38),
        (37, 99, 235),
        (16, 185, 129),
        (217, 119, 6),
        (139, 92, 246),
        (236, 72, 153),
        (14, 165, 233),
        (132, 204, 22),
    ]

    photo_boxes = []
    for i, photo in enumerate(result["photos"]):
        color = palette[i % len(palette)]
        box = _denormalize(photo["bbox"], width, height)
        draw.rectangle(box, outline=color, width=4)
        label = f"{i}:{photo.get('type', 'photo')}"
        draw.text((box[0] + 4, box[1] + 4), label, fill=color, font=font)
        photo_boxes.append((box, color))

    for assoc in result.get("associations", []):
        face_idx = assoc.get("face_idx", -1)
        if face_idx < 0 or face_idx >= len(photo_boxes):
            continue
        name_text = assoc.get("name", "").strip()
        name_bbox = None
        for n in result["names"]:
            if n.get("text", "").strip() == name_text:
                name_bbox = n.get("bbox")
                break
        if not name_bbox:
            continue
        name_box = _denormalize(name_bbox, width, height)
        face_box, color = photo_boxes[face_idx]
        draw.rectangle(name_box, outline=color, width=3)
        name_center = ((name_box[0] + name_box[2]) // 2, (name_box[1] + name_box[3]) // 2)
        face_center = ((face_box[0] + face_box[2]) // 2, (face_box[1] + face_box[3]) // 2)
        draw.line([name_center, face_center], fill=color, width=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=92)
    return output_path
