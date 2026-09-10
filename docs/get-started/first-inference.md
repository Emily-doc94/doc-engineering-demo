# Running Your First Inference

This section shows how to use the ONNX Runtime Python API to run inference with the `mnist-8.onnx` model on a handwritten digit image.

## Prerequisites

Before starting, make sure that:

- ONNX Runtime is installed. See [Installing ONNX Runtime](installation.md).
- NumPy and Pillow are installed:

```bash
pip install numpy pillow
```

## Sample Files

The example uses the following files:

| File | Description |
|---|---|
| `quickstart.py` | Complete inference script |
| `mnist-8.onnx` | ONNX model for handwritten digit recognition |
| `digit.png` | Input image used by the example |

The sample files are available in the `Examples/` directory of the project repository.

To get started, clone the repository:

```bash
git clone https://github.com/Emily-doc94/doc-engineering-demo.git
cd doc-engineering-demo
```

## Run the Example

Run the complete example from the repository root:

```bash
python examples/quickstart.py
```

The script prints the predicted digit for the sample image:

```text
Predicted digit: 6
```

The output depends on the input image. If you use a different image, the predicted digit may change.

## Understand the Code

### Load the Model

```python
import onnxruntime as ort

session = ort.InferenceSession("mnist-8.onnx")
```

`InferenceSession` loads the model and prepares the runtime session for inference.

### Preprocess the Input Image

The MNIST model expects a `[1, 1, 28, 28]` float32 tensor representing one grayscale image. Pixel values are normalized to `[0.0, 1.0]`.

```python
from PIL import Image
import numpy as np

# Read and convert to grayscale
image = Image.open("digit.png").convert("L")

# Resize to 28×28
image = image.resize((28, 28))

# Normalize to [0, 1]
img_array = np.array(image).astype(np.float32) / 255.0

# Match model input shape
input_data = img_array.reshape(1, 1, 28, 28)
```

### Run Inference

```python
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})
```

The script gets the model's input name dynamically and uses it as the key in the input dictionary.

- The first argument, `None`, requests all model outputs.
- The second argument maps the model input name to the input tensor.

### Get the Prediction

```python
predicted_digit = np.argmax(outputs[0])
print("Predicted digit:", predicted_digit)
```

For this classification model, `np.argmax()` returns the index of the highest output score, which corresponds to the predicted digit.