from flask import Flask, render_template, request, redirect, url_for
from database import init_db, save_message  # Import database functions
from export_to_excel import save_to_excel  # Import the save_to_excel function
import cv2
import numpy as np
import pytesseract
import tempfile
import base64
import time
from openpyxl import Workbook, load_workbook
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize SQLite database
init_db()

# Load TensorFlow Lite model for object detection
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = ['Book', 'Title', 'newspaper', 'laptop', 'book', 'heading', 'Title', 'trafficlight', 'bike', 'car', 
          'crosswalk', 'speedlimit', 'stop', 'invoice', 'buisness_name', 'buyer', 'seller', 'date', 'person', 
          'aeroplane', 'tvmonitor', 'train', 'dog', 'chair', 'bird', 'bottle', 'sheep', 'diningtable', 'horse', 
          'motorbike', 'sofa', 'cow', 'bicycle', 'cat', 'boat', 'bus', 'pottedplant']

# Function to process the uploaded image for object detection and OCR
def process_image(image):
    # Preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))  # Resize image to match model input size
    input_data = (np.float32(image_resized) - 127.5) / 127.5
    input_data = np.expand_dims(input_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    output_boxes = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Initialize variables for OCR
    ocr_text = ""

    # Process each detection
    for i in range(num_detections):
        class_id = int(output_classes[0][i])
        score = float(output_scores[0][i])
        if score > 0.5:  # Adjust score threshold as needed
            ymin, xmin, ymax, xmax = output_boxes[0][i]
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[class_id]}: {score:.2f}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Perform OCR using pytesseract on titles only
            if labels[class_id] == 'Title':
                title_image = image[ymin:ymax, xmin:xmax]
                title_text = pytesseract.image_to_string(title_image)
                ocr_text += f"{title_text}\n"

    return image, ocr_text

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    start_time = time.time()
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        annotated_image, ocr_text = process_image(image)
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
