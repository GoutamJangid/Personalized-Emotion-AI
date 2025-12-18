import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
import urllib.request

# --- CONFIGURATION ---
MODEL_PATH = "models/custom_final.pth"  # Pointing to my 99% model
face_cascade_path = "haarcascade_frontalface_default.xml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# ---------------------

def get_inference_model():
    """Loads the ResNet18 architecture and your 99% weights."""
    print("🔧 Loading the Super-Brain...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit()
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (Freezes Dropout/BatchNorm)
    print("✅ Model Loaded!")
    return model

def download_cascade():
    """Downloads the Haar Cascade if missing."""
    if not os.path.exists(face_cascade_path):
        print("⬇️  Downloading Face Detection Cascade...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, face_cascade_path)
        print("✅ Download Complete.")

def preprocess_face(face_img):
    """
    Exact same pipeline as Training:
    Grayscale(3ch) -> Resize(224) -> Normalize(ImageNet)
    """
    # Convert from BGR (OpenCV) to RGB (PIL/PyTorch standard)
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(face_rgb)
    return tensor.unsqueeze(0).to(DEVICE) # Add batch dimension (1, 3, 224, 224)

def main():
    download_cascade()
    model = get_inference_model()
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🚀 System Online. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to Grayscale for Face Detection (Detection is faster on gray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # 1. Extract the Face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w]
            
            # 2. Preprocess & Predict
            try:
                input_tensor = preprocess_face(face_roi)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    score, predicted = torch.max(outputs, 1)
                    
                    class_name = CLASSES[predicted.item()]
                    confidence = probabilities[0][predicted.item()].item() * 100

                # 3. Draw Results
                # Color based on emotion (Happy=Green, Angry=Red, etc.)
                color = (0, 255, 0) if class_name == 'happy' else (0, 0, 255)
                if class_name == 'neutral': color = (255, 255, 0)

                # Draw Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw Text with background strip for readability
                label = f"{class_name}: {confidence:.1f}%"
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            except Exception as e:
                pass # Skip frames where face crop is weird

        cv2.imshow('Final Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
