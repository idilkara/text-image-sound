from flask import Flask, request, jsonify, send_file, send_from_directory
from generate_image import generate_image
from generate_sound import png_to_music
from flask_cors import CORS
import io
from PIL import Image
import torch
import os
import json
import traceback
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import base64

# MongoDB configuration
MONGO_URI = "mongodb://mongodb:27017"  # Update this if your MongoDB is running on a different host/port
DB_NAME = "image_sound_db"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
images_collection = db["images"]

# Create TTL index for images collection (once at the start)
images_collection.create_index(
    [("created_at", 1)],  # Index on 'created_at' field
    expireAfterSeconds=300,  # Documents will be deleted 300 seconds (5 minutes) after 'created_at'

    # dont expire if saved 
    partialFilterExpression={"saved": False}  # Only apply TTL to documents with status 'completed'
)


print("TTL indexes created. Documents will be deleted after 5 minutes.")


app = Flask(__name__)
# Configure CORS to allow requests from any origin in development
# In production, you should specify the exact frontend URL
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # In production, replace with your frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Store temporary file paths
temp_files = {}

@app.route('/')
def serve_test():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'test.html')

@app.route('/generate-image', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        model_name = data.get('model_name', 'stabilityai/sd-turbo')
        
        # Check if CUDA is available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Allow override through request if needed
        device = data.get('device', device)

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Generate the image
        image = generate_image(prompt, model_name, device)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Save to MongoDB
        image_doc = {
            'prompt': prompt,
            'image_data': img_byte_arr,
            'created_at': datetime.now(timezone.utc),
            'saved': False 
        }
        
        result = images_collection.insert_one(image_doc)
        image_id = str(result.inserted_id)

        # Return both the image and its ID
        return jsonify({
            'image_id': image_id,
            'image_data': base64.b64encode(img_byte_arr).decode('utf-8')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-sound', methods=['POST'])
def generate_sound():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        max_length_seconds = data.get('max_length_seconds', 60)
        image_id = data.get('image_id')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Get image
        image_doc = images_collection.find_one({'_id': ObjectId(image_id)})
        if not image_doc:
            return jsonify({'error': 'Image not found'}), 404


        sound_id = image_id

        # Generate sound in memory
        sound_io = io.BytesIO()
        png_to_music(io.BytesIO(image_doc['image_data']), sound_io, max_length_seconds)

        # Seek to the beginning of the byte stream before returning the response
        sound_io.seek(0)

        # Save sound to MongoDB to image collection - update the image document

        images_collection.update_one(
            {'_id': ObjectId(image_id)},
            {'$set': {
                'sound_data': sound_io.getvalue(),     
            }}
        )

        # Return the sound file as a response with correct content type
        return send_file(
            sound_io,  # Streamed byte data
            mimetype='audio/wav',  # WAV format
            as_attachment=True,  # Force download of the file
            download_name='generated_sound.wav'
        )
        
    except Exception as e:
        print(f"Error in generate_sound endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/save-image/<image_id>', methods=['POST'])
def save_image(image_id):

    try:
        # Find the image document by ID
        image_doc = images_collection.find_one({'_id': ObjectId(image_id)})
        if not image_doc:
            return jsonify({'error': 'Image not found'}), 404

        # Update the document to mark it as saved
        images_collection.update_one(
            {'_id': ObjectId(image_id)},
            {'$set': {'saved': True}}
        )
        return jsonify({'message': 'Image saved successfully'}), 200

    except Exception as e:
        print(f"Error in save_image endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



@app.route('/getImages', methods=['GET'])
def get_images():
    try:
        # Fetch all images from the database
        images = list(images_collection.find({}, {
            '_id': 1, 
            'prompt': 1, 
            'created_at': 1, 
            'sound_data': 1,
            'image_data': 1,
            'saved': 1
        }))

        # Convert ObjectId to string and format datetime
        for image in images:
            image['_id'] = str(image['_id'])
            image['created_at'] = image['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert binary image data to base64 for JSON response
            if 'image_data' in image and image['image_data']:
                image['image_data'] = base64.b64encode(image['image_data']).decode('utf-8')
                
            # Convert binary sound data to base64 for JSON response if it exists
            if 'sound_data' in image and image['sound_data']:
                image['sound_data'] = base64.b64encode(image['sound_data']).decode('utf-8')
        
        return jsonify(images), 200

    except Exception as e:
        print(f"Error in get_images endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
