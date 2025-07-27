import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class MaskDetector:
    def __init__(self, face_detector_prototxt, face_detector_weights, mask_model_path):
        """Initialize the face mask detector with model paths"""
        self.face_net = cv2.dnn.readNet(face_detector_prototxt, face_detector_weights)
        self.mask_net = load_model(mask_model_path)
        self.confidence_threshold = 0.5
        
    def detect_and_predict_mask(self, frame, try_on_mask=None):
        """
        Detect faces in frame and predict mask status
        Optionally apply virtual mask overlay if try_on_mask is provided
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        locs = []
        preds = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes stay within frame dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                
                # Apply virtual mask if provided
                if try_on_mask is not None:
                    self._apply_mask(frame, startX, startY, endX, endY, try_on_mask)
                
                # Prepare face for mask detection
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.mask_net.predict(faces, batch_size=32)
            
        return locs, preds
    
    def _apply_mask(self, frame, startX, startY, endX, endY, mask_img):
        """Apply virtual mask overlay to face"""
        face_w = endX - startX
        face_h = endY - startY
        
        # Resize mask to fit face
        mask = cv2.resize(mask_img, (face_w, face_h))
        
        # Extract alpha channel if exists
        if mask.shape[2] == 4:
            alpha = mask[:, :, 3] / 255.0
            inv_alpha = 1.0 - alpha
            mask = mask[:, :, :3]
        else:
            alpha = 0.7  # Default opacity if no alpha channel
            inv_alpha = 1.0 - alpha
        
        # Blend mask with face
        for c in range(0, 3):
            frame[startY:endY, startX:endX, c] = (
                alpha * mask[:, :, c] + 
                inv_alpha * frame[startY:endY, startX:endX, c]
            )

def main():
    # Initialize detector
    detector = MaskDetector(
        face_detector_prototxt="face_detector/deploy.prototxt",
        face_detector_weights="face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        mask_model_path="mask_detector.model"
    )
    
    # Load virtual mask for try-on
    try_on_mask = cv2.imread("static/masks/surgical.png", cv2.IMREAD_UNCHANGED)
    
    # Start video stream
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect masks and optionally apply virtual mask
        locs, preds = detector.detect_and_predict_mask(frame, try_on_mask=try_on_mask)
        
        # Draw bounding boxes and labels
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, without_mask) = pred
            
            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = f"{label}: {max(mask, without_mask) * 100:.2f}%"
            
            cv2.putText(frame, label, (startX, startY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        cv2.imshow("Virtual Mask Try-On", frame)
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('m'):  # Toggle mask
            try_on_mask = None if try_on_mask is not None else cv2.imread(
                "static/masks/surgical.png", cv2.IMREAD_UNCHANGED)
        elif key == ord('s'):  # Save snapshot
            cv2.imwrite("snapshot.jpg", frame)
            print("Snapshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()