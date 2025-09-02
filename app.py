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
import cv2

from ultralytics import YOLO  # pip install ultralytics
from google import genai      # pip install google-genai
from google.genai import types
from dotenv import load_dotenv


# ------------------------
# Config
# ------------------------
# Expect GOOGLE_API_KEY in env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Set it before running or Gemini calls will fail.")

app = Flask(__name__)
CORS(app)

# YOLOv8s (downloaded automatically on first run)
model = None

def get_yolo_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")  
    return model

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
    """
    Form-data:
      - image: file
      - question (optional): string (VQA question)
    Returns JSON:
      - detections: [{label, confidence, box:[x1,y1,x2,y2]}]
      - annotated_image_b64: base64 JPEG (no prefix)
      - answer: VQA answer string
    """
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "invalid or unsupported file"}), 400

    question = request.form.get("question", "").strip()
    model = get_yolo_model()

    raw = f.read()
    pil = pil_from_bytes(raw)
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # --- YOLOv8s inference ---
    res = model.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)[0]

    detections = []
    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy()[0].tolist()  # [x1,y1,x2,y2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        conf = float(b.conf.cpu().numpy()[0])
        cls = int(b.cls.cpu().numpy()[0])
        label = model.names[cls] if hasattr(model, "names") else str(cls)

        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

    annotated = draw_boxes(pil, detections)
    annotated_b64 = img_to_b64(annotated)

    # --- Build VQA prompt ---
    # We pass both the detections summary and the user's question to Gemini, alongside the image itself.
    summary_lines = [f"- {d['label']} (conf {d['confidence']:.2f}) at {d['box']}" for d in detections] or ["- No objects detected"]
    det_summary = "Objects detected by YOLO:\n" + "\n".join(summary_lines)

    # user_q = question if question else "Provide a concise description and count of key objects."
    prompt_text = (
        """You are an expert saftey auditor assistant.  
        Given an image, analyze it and return the findings which are a threat to Human Life or violate the general industrial and residential saftey standards strictly in JSON format.  

        Image context: {det_summary}

        The JSON must contain exactly 3 keys: "wrong", "right", and "todo".  

        Each should be an array of objects in the following formats:

        - wrong = { "heading": <short title>, "severity": <number from 1 to 10>, "explanation": <detailed text> }
        - right = { "heading": <short title> }
        - todo  = { "heading": <short action point> }

        Rules:
        1. Always include up to 10 points per section. If fewer, return only available points.
        2. "severity" must always be an integer between 1 and 10 (10 = most severe).
        3. Keep "heading" short (max 5 words).
        4. Explanations must be 1–3 sentences max.
        5. Do not include anything outside the JSON object. No markdown, no prose.

        Example valid JSON response:
        {
        "wrong": [
            {"heading": "Overloaded Wire", "severity": 3, "explanation": "The wire gauge is too thin for the load, causing overheating risk."}
        ],
        "right": [
            {"heading": "Proper Grounding"}
        ],
        "todo": [
            {"heading": "Replace thin wire with proper gauge."}
        ]
        }
        """
    )

    # --- Gemini multimodal call (image + text) ---
    answer_text = ""
    try:
        # The Python client accepts PIL.Image as a content item in many versions.
        # If your version requires explicit parts, adapt accordingly.
        g_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil, prompt_text]
        )
        # .text is available on successful responses
        answer_text = getattr(g_response, "text", "") or ""
    except Exception as e:
        # If Gemini fails, still return detections and annotated image
        answer_text = f"(VQA error) {e}"

    try:
        # Clean response (Gemini sometimes wraps JSON in markdown ```json ... ```)
        cleaned = answer_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].replace("json", "", 1).strip()
        
        analysis = json.loads(cleaned)  # must be a dict with wrong/right/todo
    except Exception as e:
        return jsonify({"error": f"Invalid response from Gemini: {str(e)}"})
    
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
