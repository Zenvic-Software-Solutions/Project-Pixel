from flask import Flask, request, render_template, jsonify ,send_from_directory
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
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    if not encodings:
        return jsonify({"status": "error", "message": "No face detected in the image."})

    conn = sqlite3.connect("face_database.db")
    cursor = conn.cursor()

    stored_faces = []
    for i, encoding in enumerate(encodings):
        encoding_data = encoding.tobytes()
        
        # ✅ Keep original filename but index multiple faces
        face_indexed_filename = f"{file.filename}"
        
        # ✅ Store encoding in database
        cursor.execute("INSERT INTO faces (filename, encoding) VALUES (?, ?)", (face_indexed_filename, encoding_data))
        stored_faces.append(face_indexed_filename)

    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": f"{len(encodings)} faces detected and stored!", "faces": stored_faces})


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
        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "No image data received."})

        webcam_image = decode_base64_image(data["image"])
        if webcam_image is None:
            return jsonify({"status": "error", "message": "Invalid image format."})

        # Convert to RGB for face recognition processing
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)

        # Detect faces and extract their encodings
        face_locations = face_recognition.face_locations(webcam_image_rgb)
        face_encodings = face_recognition.face_encodings(webcam_image_rgb, face_locations)

        if not face_encodings:
            return jsonify({"status": "error", "message": "No face detected in the image."})

        # Fetch stored face encodings from the database
        conn = sqlite3.connect("face_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT filename, encoding FROM faces")
        stored_faces = cursor.fetchall()
        conn.close()

        matched_faces = []
        
        for stored_filename, encoding_blob in stored_faces:
            stored_encoding = np.frombuffer(encoding_blob, dtype=np.float64)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
                distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]

                similarity = round((1 - distance) * 100, 2)  # Convert similarity to percentage

                if match[0] and similarity >= 50:  # ✅ Only include faces with at least 50% similarity
                    matched_faces.append({
                        "file": stored_filename,
                        "similarity": similarity,
                        "face_location": face_location
                    })

        if matched_faces:
            matched_faces.sort(key=lambda x: x["similarity"], reverse=True)
            return jsonify({"status": "success", "matches": matched_faces})

        return jsonify({"status": "error", "message": "No matching faces found above 50% similarity."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing image: {str(e)}"})


    
# ✅ Route to serve images
@app.route('/dataset/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
