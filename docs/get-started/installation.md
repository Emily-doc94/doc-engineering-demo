# Installing ONNX Runtime

This section explains how to install ONNX Runtime for Python applications and verify the installation.

## Prerequisites

Before installing ONNX Runtime, make sure that:

- Python is installed and available in your environment.
- Your Python version is supported by the ONNX Runtime release you plan to install.
- `pip` is available for installing Python packages.

For the latest Python version requirements and environment compatibility information, see the [ONNX Runtime compatibility documentation](https://onnxruntime.ai/docs/reference/compatibility.html).

## Install ONNX Runtime

ONNX Runtime provides different Python packages for different execution environments.

| Package | Use case |
|---|---|
| `onnxruntime` | Run inference on the CPU |
| `onnxruntime-gpu` | Run inference with supported GPU execution providers |

For the examples in this guide, use the CPU package. It provides the simplest setup for learning the ONNX Runtime Python API.

> **Note:** Do not install `onnxruntime` and `onnxruntime-gpu` in the same Python environment.

### Install the CPU Package

```bash
pip install onnxruntime
```

### Install the GPU Package

If your application requires GPU acceleration, install the GPU package instead:

```bash
pip install onnxruntime-gpu
```

GPU execution requires additional dependencies and a compatible execution provider. For GPU-specific requirements, see the [ONNX Runtime installation documentation](https://onnxruntime.ai/docs/install/).

## Verify the Installation

After installing ONNX Runtime, verify that the package can be imported successfully.

```python
import onnxruntime as ort

print(ort.__version__)
```

If the installation is successful, the command prints the installed ONNX Runtime version.
