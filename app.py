from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Constants
UPLOAD_FOLDER = 'static/masks'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models at startup
face_net = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
mask_detector = load_model("mask_detector.model")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_predict_mask(frame, face_net, mask_net, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding boxes stay within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = Image.fromarray(face)
            face = np.array(face) / 255.0
            face = np.expand_dims(face, axis=0)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        preds = mask_net.predict(np.vstack(faces))
    
    return (locs, preds)

@app.route("/detect_mask", methods=["POST"])
def detect_mask():
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image provided"}), 400
            
        data = request.json["image"]
        if not data.startswith('data:image'):
            return jsonify({"error": "Invalid image format"}), 400
            
        # Process image
        header, encoded = data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect faces and masks
        (locs, preds) = detect_and_predict_mask(frame, face_net, mask_detector)
        
        results = []
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask_prob, no_mask_prob) = pred
            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            probability = float(max(mask_prob, no_mask_prob))
            
            results.append({
                "box": [int(startX), int(startY), int(endX), int(endY)],
                "label": label,
                "probability": probability,
                "mask_prob": float(mask_prob),
                "no_mask_prob": float(no_mask_prob)
            })
            
        return jsonify({
            "results": results,
            "count": len(results),
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/try_mask", methods=["POST"])
def try_mask():
    try:
        # Get image and selected mask type
        data = request.json
        image_data = data["image"]
        mask_type = data.get("mask_type", "surgical")
        
        # Decode image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        (locs, _) = detect_and_predict_mask(frame, face_net, mask_detector)
        
        if not locs:
            return jsonify({"error": "No faces detected"}), 400
            
        # Load mask overlay
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{mask_type}.png")
        if not os.path.exists(mask_path):
            return jsonify({"error": "Mask type not available"}), 404
            
        mask_overlay = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Apply mask to each face
        for (startX, startY, endX, endY) in locs:
            face_w = endX - startX
            face_h = endY - startY
            
            # Resize mask to fit face
            resized_mask = cv2.resize(mask_overlay, (face_w, face_h))
            
            # Extract alpha channel
            mask_alpha = resized_mask[:, :, 3] / 255.0
            inv_mask_alpha = 1.0 - mask_alpha
            
            # Blend mask with face
            for c in range(0, 3):
                frame[startY:endY, startX:endX, c] = (
                    mask_alpha * resized_mask[:, :, c] +
                    inv_mask_alpha * frame[startY:endY, startX:endX, c]
                )
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_result = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "result_image": f"data:image/jpeg;base64,{encoded_result}",
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/upload_mask', methods=['POST'])
def upload_mask():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({
            "message": "Mask uploaded successfully",
            "filename": filename
        }), 200
        
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/masks/<filename>')
def get_mask(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)