from flask import Flask, render_template, request, redirect, url_for
from database import init_db, save_message  # Import database functions
from export_to_excel import save_to_excel  # Import the save_to_excel function
import cv2
import numpy as np
import tempfile
import base64
import time
from openpyxl import Workbook, load_workbook
import tensorflow as tf
from paddleocr import PaddleOCR, draw_ocr
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize SQLite database
init_db()

models = {
    "car_bike": {
        "path": "D:\\p2\\converted_model.tflite",
        "labels": ['Car', 'Bike', 'Other']
    },
    "book_title": {
        "path": "D:\\p2\\detect.tflite",
        "labels": ['Book', 'Title']
    }
}

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Function to process the uploaded image for object detection and OCR using PaddleOCR
def process_image(image, model_info):

    # Load TFLite model and allocate tensors (similar to your existing code)
    interpreter = tf.lite.Interpreter(model_path=model_info["path"])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    input_data = (np.float32(image_resized) - 127.5) / 127.5
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_boxes = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    ocr_text = ""

    for i in range(num_detections):
        class_id = int(output_classes[0][i])
        score = float(output_scores[0][i])
        if score > 0.65:
            ymin, xmin, ymax, xmax = output_boxes[0][i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = model_info["labels"][class_id]
            cv2.putText(image, f"{label}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if model_info["labels"][class_id] == 'Title':
                title_image = image[ymin:ymax, xmin:xmax]
                result = ocr.ocr(title_image, cls=True)
                for line in result:
                    for res in line:
                        ocr_text += res[1][0] + "\n"

    return image, ocr_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/upload', methods=['POST'])
def upload():
    detection_type = request.form.get('detection_type')
    if detection_type not in models:
        return render_template('index.html', error='Invalid detection type selected')
    
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    start_time = time.time()
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        model_info = models[detection_type]
        annotated_image, ocr_text = process_image(image, model_info)
        detection_time = time.time() - start_time

        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('result.html', is_video=False, annotated_result=annotated_image_base64, ocr_text=ocr_text, detection_time=detection_time)

    else:
        return render_template('index.html', error='Unsupported file type')

@app.route('/submit_message', methods=['POST'])
def submit_message():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        try:
            # Save message to SQLite database
            save_message(name, email, message)

            # Save message to Excel file
            save_to_excel(name, email, message)

            # Redirect to success page
            return redirect(url_for('submit_message_success'))

        except Exception as e:
            print(f"Error saving message: {str(e)}")
            return render_template('index.html', error='Error submitting message')

    # Redirect to index if GET request
    return redirect(url_for('index'))

@app.route('/submit_message/success')
def submit_message_success():
    return render_template('submit_message.html')

if __name__ == '__main__':
    app.run(debug=True)
