import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import base64

app = Flask(__name__)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/ratnakar/project/p1/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = ['Book', 'Title']

# Function to process the uploaded image for object detection
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

    # Draw bounding boxes on the image
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
            label = labels[class_id]
            cv2.putText(image, f"{label}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        annotated_image = process_image(image)

        # Encode annotated image as base64 string
        _, annotated_img_encoded = cv2.imencode('.jpg', annotated_image)
        annotated_img_base64 = base64.b64encode(annotated_img_encoded).decode('utf-8')

        return render_template('result.html', annotated_image=annotated_img_base64)

    return render_template('index.html', error='File not processed')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
