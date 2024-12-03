from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load your trained CIFAR-10 model
model = load_model('cifar10_model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs (name, email, and feedback)
    user_name = request.form.get('user_name')
    user_email = request.form.get('user_email')
    feedback = request.form.get('feedback')

    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    # Convert the file to a BytesIO object and open the image
    img = Image.open(BytesIO(file.read()))

    # Resize image to match the model input size (32x32 for CIFAR-10)
    img = img.resize((32, 32))

    # Preprocess image for prediction (convert to array and normalize)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make the prediction
    prediction = model.predict(img_array)

    # Get the predicted class index (the class with the highest probability)
    predicted_class = np.argmax(prediction, axis=1)

    # Get the class name based on the index
    predicted_class_name = class_names[predicted_class[0]]

    # Return the prediction result and user inputs to the HTML page
    return render_template('index.html', prediction=predicted_class_name, user_name=user_name, feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
