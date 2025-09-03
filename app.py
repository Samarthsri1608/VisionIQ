import os
import io
import base64
from datetime import datetime
from typing import List, Dict, Any
import json

from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from google import genai      # pip install google-genai
from google.genai import types
from dotenv import load_dotenv


# ------------------------
# Config
# ------------------------
# Expect GOOGLE_API_KEY in env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Set it before running or Gemini calls will fail.")

app = Flask(__name__)
CORS(app)

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


# ------------------------
# Helpers
# ------------------------
def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def img_to_b64(pil_img: Image.Image) -> str:
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Pillow 10+ removed textsize; use textbbox for compatibility."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        # Fallback for very old Pillow
        return draw.textsize(text, font=font)

def draw_boxes(pil_img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['confidence']:.2f}"

        # box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # label background
        tw, th = text_size(draw, label, font)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill="red")
        # text
        draw.text((x1 + 4, y1 - th - 4), label, fill="white", font=font)

    return img


# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    # Serve the frontend if placed alongside app.py
    return send_file("template/index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "invalid or unsupported file"}), 400

    raw = f.read()
    pil = pil_from_bytes(raw)

    w, h = pil.size

    # --- Prompt for Gemini ---
    prompt_text = f"""
        You are an expert safety auditor assistant.

        Task:
        Analyze the uploaded image ({w}x{h} pixels) and identify hazards that pose a risk to human life
        or violate general safety standards. Return ONLY a valid JSON object.

        JSON specification:
        {{
        "wrong": [
            {{
            "heading": "short title (≤5 words)",
            "severity": <integer 1–5>,
            "explanation": "1–3 sentences",
            "box": [x1, y1, x2, y2]   // absolute pixel coordinates for this image
            }}
        ],
        "right": [
            {{
            "heading": "short title"
            }}
        ],
        "todo": [
            {{
            "heading": "short action point"
            }}
        ]
        }}

        Important rules:
        1. Do not return markdown, comments, or text outside the JSON.
        2. All bounding boxes MUST use absolute pixel coordinates, where:
        - (0,0) = top-left corner,
        - (w,h) = bottom-right corner,
        - image size = {w} pixels wide × {h} pixels tall.
        3. If no hazards are found, return "wrong": [].
        4. Return up to 10 items per section (wrong/right/todo).
    """


    # --- Gemini API call ---
    try:
        g_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil, prompt_text]
        )
        answer_text = getattr(g_response, "text", "") or ""
    except Exception as e:
        return jsonify({"error": f"Gemini request failed: {str(e)}"}), 500

    # --- Parse JSON safely ---
    try:
        cleaned = answer_text.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ```
            cleaned = cleaned.split("```")[1].replace("json", "", 1).strip()
        analysis = json.loads(cleaned)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON from Gemini: {str(e)}", "raw": answer_text}), 500

    # --- Build detections list from "wrong" ---
    detections = []
    for w_item in analysis.get("wrong", []):
        box = w_item.get("box")
        if box and len(box) == 4:
            # Ensure ints
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append({
                "label": w_item.get("heading", "Hazard"),
                "confidence": w_item.get("severity", 0),
                "box": [x1, y1, x2, y2]
            })

    # --- Draw annotated image ---
    annotated = draw_boxes(pil, detections)
    annotated_b64 = img_to_b64(annotated)

    return jsonify({
        "detections": detections,
        "annotated_image_b64": annotated_b64,
        "wrong": analysis.get("wrong", []),
        "right": analysis.get("right", []),
        "todo": analysis.get("todo", [])
    })




if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=os.getenv("PORT",""), debug=True)
