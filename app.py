from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-platform access

# Global placeholders for lazy loading
processor = None
model = None

def load_model():
    """Lazy-load BLIP model only when needed to reduce startup memory usage."""
    global processor, model
    if processor is None or model is None:
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Model loaded successfully!")


@app.route('/caption', methods=['POST'])
def caption_image():
    """Generate and optionally translate an image caption."""
    try:
        print("Received caption request...")
        data = request.json

        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_base64 = data['image']
        language = data.get('language', 'en')

        print(f"Processing image for language: {language}")

        # Decode image from base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Load model (lazy initialization)
        load_model()

        # Generate caption
        text = "You are seeing a"
        inputs = processor(image, text, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"Generated caption: {caption}")

        # Translate if language != English
        translated = caption
        if language != 'en':
            try:
                translated = GoogleTranslator(source='en', target=language).translate(caption)
                print(f"Translated caption: {translated}")
            except Exception as e:
                print(f"Translation error: {e}")
                # Fallback to English if translation fails

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
    """Simple health-check route."""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})


if __name__ == '__main__':
    # Bind to Render's dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
