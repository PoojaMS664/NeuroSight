from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile access

# Load model once at startup
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model loaded successfully!")

@app.route('/caption', methods=['POST'])
def caption_image():
    try:
        print("Received request...")
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_base64 = data['image']
        language = data.get('language', 'en')
        
        print(f"Processing image for language: {language}")
        
        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Generate caption
        text = "You are seeing a"
        inputs = processor(image, text, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"Generated caption: {caption}")
        
        # Translate if not English
        translated = caption
        if language != 'en':
            try:
                translated = GoogleTranslator(source='en', target=language).translate(caption)
                print(f"Translated: {translated}")
            except Exception as e:
                print(f"Translation error: {e}")
                # Fall back to English if translation fails
        
        return jsonify({
            'caption': caption,
            'translated': translated,
            'language': language,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

