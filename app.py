from flask import Flask, request, jsonify
from flask_cors import CORS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
import json
import traceback
import google.generativeai as genai  

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# ✅ Use free-tier model (instead of "gemini-pro")
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

app = Flask(__name__)
CORS(app)

Load models
try:
    plant_model = tf.keras.models.load_model("/home/ubuntu/Medi-Backend/models/my_modelp.keras")
    leaf_model = tf.keras.models.load_model("/home/ubuntu/Medi-Backend/models/my_modell.keras")
    print("Models loaded.")
except Exception as e:
    print("Model loading error:", e)
    plant_model = None
    leaf_model = None
# import os
# import gdown

# PLANT_MODEL_PATH = "my_modelp.keras"
# LEAF_MODEL_PATH = "my_modell.keras"

# PLANT_MODEL_URL = "https://drive.google.com/uc?id=15F883O011XGyC_8HS3Xe-IX8_B6xu8Ss"
# LEAF_MODEL_URL = "https://drive.google.com/uc?id=13rHVQ7A9i_qfJkfU_lkMMJtav_a2avCW"

# def download_model_if_needed(path, url):
#     if not os.path.exists(path):
#         print(f"Downloading model: {path} ...")
#         try:
#             gdown.download(url, path, quiet=False)
#             print(f"Downloaded {path}")
#         except Exception as e:
#             print(f" Failed to download model from {url}")
#             raise e

# try:
#     download_model_if_needed(PLANT_MODEL_PATH, PLANT_MODEL_URL)
#     download_model_if_needed(LEAF_MODEL_PATH, LEAF_MODEL_URL)

#     plant_model = tf.keras.models.load_model(PLANT_MODEL_PATH)
#     leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH)
#     print(" Models loaded successfully.")

# except Exception as e:
#     print("Model loading error:", e)
#     plant_model = None
#     leaf_model = None


# Load plant and leaf data
try:
    with open("plantinfo.json", 'r', encoding='utf-8') as f:
        plant_data = json.load(f)
    with open("leavesinfo.json", 'r', encoding='utf-8') as f:
        leaf_data = json.load(f)
except Exception as e:
    print("Data loading error:", e)
    plant_data = []
    leaf_data = []

# Image preprocessing
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print("Preprocessing error:", e)
        return None

# 🔍 Search and filter endpoint
@app.route("/search", methods=["GET"])
def search_plants():
    name_query = request.args.get("name", "").lower()
    usage_query = request.args.get("usage", "").lower()

    filtered = []

    for plant in plant_data:
        name_match = name_query in plant.get("Name", "").lower()
        usage_match = usage_query in plant.get("Medical_Usage", "").lower()
        
        if (not name_query or name_match) and (not usage_query or usage_match):
            filtered.append(plant)

    return jsonify(filtered)

# 🔮 Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files or 'uploadType' not in request.form:
        return jsonify({'error': 'File or uploadType not provided'}), 400

    file = request.files['file']
    upload_type = request.form['uploadType']
    processed_image = preprocess_image(file.read())

    if processed_image is None:
        return jsonify({'error': 'Failed to process image'}), 400

    try:
        if upload_type == "Plant" and plant_model:
            prediction = plant_model.predict(processed_image)
            index = int(np.argmax(prediction))
            result = plant_data[index] if index < len(plant_data) else {"error": "Plant not found"}

        elif upload_type == "Leaf" and leaf_model:
            prediction = leaf_model.predict(processed_image)
            index = int(np.argmax(prediction))
            result = leaf_data[index] if index < len(leaf_data) else {"error": "Leaf not found"}

        else:
            return jsonify({'error': 'Invalid uploadType or model not loaded'}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# 💬 Gemini chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message")

        if not user_msg:
            return jsonify({"error": "No message provided"}), 400

        print("User message:", user_msg)

        response = gemini_model.generate_content(
            f"You are a helpful assistant specialized in medicinal plants. {user_msg}"
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        print("Chat error:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
from threading import Lock

votes_file = "votes.json"
votes_lock = Lock()

# Helper to read votes from file
def load_votes():
    try:
        with open(votes_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Helper to write votes to file
def save_votes(votes):
    with open(votes_file, "w", encoding="utf-8") as f:
        json.dump(votes, f, indent=2, ensure_ascii=False)

# ✅ Get votes for a plant
@app.route("/votes/<plant_name>", methods=["GET"])
def get_votes(plant_name):
    with votes_lock:
        votes = load_votes()
    plant_votes = votes.get(plant_name, {"upvotes": 0, "downvotes": 0})
    return jsonify(plant_votes)

# ✅ Upvote or downvote
@app.route("/vote", methods=["POST"])
def vote():
    data = request.get_json()
    plant_name = data.get("plant_name")
    vote_type = data.get("type")  # "upvote" or "downvote"

    if not plant_name or vote_type not in ["upvote", "downvote"]:
        return jsonify({"error": "Invalid request"}), 400

    with votes_lock:
        votes = load_votes()
        if plant_name not in votes:
            votes[plant_name] = {"upvotes": 0, "downvotes": 0}

        if vote_type == "upvote":
            votes[plant_name]["upvotes"] += 1
        else:
            votes[plant_name]["downvotes"] += 1

        save_votes(votes)

    return jsonify({"message": f"{vote_type.capitalize()} recorded for {plant_name}."})
@app.route('/update-history', methods=['POST'])
def update_history():
    new_entry = request.json
    with open("history.json", "r+") as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                data = []
        except:
            data = []

        data.append(new_entry)
        file.seek(0)
        json.dump(data, file, indent=2)
        file.truncate()
    return jsonify({"status": "success"})

@app.route('/history', methods=['GET'])
def get_history():
    with open("history.json", "r") as file:
        try:
            data = json.load(file)
        except:
            data = []
    return jsonify(data)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    with open("history.json", "w") as file:
        json.dump([], file)
    return jsonify({"status": "cleared"})

# 🏁 Run Flask app
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    app.run(debug=True, host="0.0.0.0", port=5000)
