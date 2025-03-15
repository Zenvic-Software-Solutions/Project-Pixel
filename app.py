from flask import Flask, request, render_template, jsonify
import cv2
import face_recognition
import numpy as np
import base64
import os
import re
import binascii
import sqlite3
import base64

# ✅ Function to decode base64 image to OpenCV format
def decode_base64_image(image_base64):
    try:
        # 🔹 Remove the 'data:image/jpeg;base64,' or similar prefix
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # 🔹 Decode Base64
        image_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)

        # 🔹 Convert to OpenCV Image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Decoded image is None")

        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
    

app = Flask(__name__)

UPLOAD_FOLDER = "dataset/image/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_valid_base64(base64_string):
    try:
        base64.b64decode(base64_string, validate=False)  # ✅ Corrected
        return True
    except binascii.Error:
        return False

# ✅ Route 1: Upload Image Page
@app.route("/")
def upload_page():
    return render_template("upload.html")

# ✅ Route 2: Handle Image Upload

# ✅ Create a SQLite Database and Table
conn = sqlite3.connect("face_database.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        encoding BLOB
    )
""")
conn.commit()
conn.close()

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # ✅ Detect faces and extract encodings
    image = face_recognition.load_image_file(file_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"status": "error", "message": "No face detected in the image."})

    encoding_data = encodings[0].tobytes()  # Convert encoding to binary

    # ✅ Store encoding in the database
    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (filename, encoding) VALUES (?, ?)", (file.filename, encoding_data))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "File uploaded and face data stored!"})

# ✅ Route 3: Face Detection Page
@app.route("/detect")
def detect_page():
    return render_template("detect.html")

# ✅ Function to enhance images
def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Convert back to BGR (for color models)
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced

# ✅ Route 4: Handle Face Matching with Enhancement
@app.route("/match", methods=["POST"])
def match_faces():
    try:
        data = request.json
        if "image" not in data:
            return jsonify({"status": "error", "message": "No image data received."})

        webcam_image = decode_base64_image(data["image"])
        if webcam_image is None:
            return jsonify({"status": "error", "message": "Invalid image format."})

        # ✅ Convert to RGB and extract face encodings
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(webcam_image_rgb)

        if not face_encodings:
            return jsonify({"status": "error", "message": "No face detected in the image."})

        # ✅ Fetch stored encodings from the database
        conn = sqlite3.connect("face_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT filename, encoding FROM faces")
        stored_faces = cursor.fetchall()
        conn.close()

        matched_images = []

        for filename, encoding_blob in stored_faces:
            stored_encoding = np.frombuffer(encoding_blob, dtype=np.float64)

            for face_encoding in face_encodings:
                match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
                distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]

                if match[0]:  # ✅ Match found
                    matched_images.append({"file": filename, "similarity": round(1 - distance, 2)})

        if matched_images:
            matched_images.sort(key=lambda x: x["similarity"], reverse=True)
            return jsonify({"status": "success", "matches": matched_images})

        return jsonify({"status": "error", "message": "No matching images found."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing image: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
