import warnings
from urllib3.exceptions import NotOpenSSLWarning  # Add this import

# Suppress the LibreSSL warning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Rest of your code...
from flask import Flask, render_template, request
import os
import requests
import random
import cv2
import pytesseract
import torch
import numpy as np
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image
import torchvision.models as models
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
workers = 1
threads = 1
# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
nltk.download('punkt', quiet=True)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Improved funny predictions
funny_predictions = [
    "Alert! Future billionaire detected... in Minecraft!",
    "NASA's calling... to test zero-gravity naps!",
    "Warning: Pizza overload in your near future!",
    "Career change ahead: Professional meme curator!"
]

# Load ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# User-friendly label mapping
LABEL_MAPPING = {
    "web site": "website screenshot",
    "monitor": "computer screen",
    "cellphone": "mobile device",
    "notebook": "laptop",
    "space bar": "keyboard",
    "website": "webpage",
    "screen": "digital screen"
}

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = requests.get(LABELS_URL).text.split("\n")
imagenet_labels = {i: label.strip() for i, label in enumerate(imagenet_labels) if label.strip()}

def is_screenshot(image_path):
    """Detect screen-like elements using contour analysis."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for large rectangular contours
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
                return True
        return False
    except Exception as e:
        print(f"Screenshot detection error: {e}")
        return False

# Replace the MTCNN face detection with OpenCV's Haar Cascade
def detect_face(image_path):
    """Face detection using OpenCV's Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return len(faces) > 0
    except Exception as e:
        print(f"Face detection error: {e}")
        return False

def extract_text(image_path):
    """Enhanced OCR processing for screenshots."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
            
        # Preprocessing for screen content
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
                                    
        # OCR with optimized settings
        text = pytesseract.image_to_string(gray, config='--psm 6 --oem 3')
        
        # Smart summarization
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 3:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = " ".join(str(s) for s in summarizer(parser.document, 3))
            return summary
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def classify_image(image_path):
    """Classification with user-friendly labels."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
        _, predicted = outputs.max(1)
        label = imagenet_labels.get(predicted.item(), "unknown object")
        return LABEL_MAPPING.get(label.lower(), label)
    except Exception as e:
        print(f"Classification error: {e}")
        return "unrecognized item"

def get_funny_prediction():
    return random.choice(funny_predictions)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            try:
                # Processing pipeline
                extracted_text = extract_text(file_path)
                
                if extracted_text.strip():
                    summary = f"Extracted text: {extracted_text}"
                else:
                    if is_screenshot(file_path):
                        summary = "This appears to be a screen capture"
                    elif detect_face(file_path):
                        summary = get_funny_prediction()
                    else:
                        label = classify_image(file_path)
                        summary = f"This looks like: {label.capitalize()}"
                        
            except Exception as e:
                summary = f"Error processing image: {str(e)}"
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
