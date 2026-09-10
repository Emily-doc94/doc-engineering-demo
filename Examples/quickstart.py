import numpy as np
from PIL import Image
import onnxruntime as ort


def main():
    # Load the model
    session = ort.InferenceSession("mnist-8.onnx")

    # Preprocess the input image
    # Read and convert to grayscale
    image = Image.open("digit.png").convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Normalize pixel values from 0-255 to 0.0-1.0
    img_array = np.array(image).astype(np.float32) / 255.0

    # Match model input shape: [batch, channel, height, width]
    input_data = img_array.reshape(1, 1, 28, 28)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    # Get predicted digit
    predicted_digit = np.argmax(outputs[0])
    print("Predicted digit:", predicted_digit)


if __name__ == "__main__":
    main()