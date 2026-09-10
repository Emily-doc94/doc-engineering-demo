 InferenceSession API Reference

This chapter provides reference information for the `InferenceSession` class, the primary Python interface for running inference with ONNX Runtime.

## Overview

`InferenceSession` loads an ONNX model into memory and exposes methods for inspecting model metadata and executing inference.

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
```

A session can be reused for multiple `run()` calls without reloading the model.

---

## Constructor

```python
ort.InferenceSession(
    path_or_bytes,
    sess_options=None,
    providers=None,
)
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path_or_bytes` | `str` or `bytes` | Yes | Path to the ONNX model file, or serialized model bytes. |
| `sess_options` | `SessionOptions` | No | Session-level configuration such as thread count and graph optimization settings. |
| `providers` | `list[str]` | No | Execution providers to use, specified in priority order. If omitted, ONNX Runtime uses the available providers with its default precedence. |

---

## Methods

### `get_inputs()`

Returns metadata for all model inputs.

```python
inputs = session.get_inputs()
```

**Returns:** A list of `NodeArg` objects.

| Attribute | Description |
|---|---|
| `name` | Input node name. Use this as the key in `run()`. |
| `shape` | Expected tensor dimensions, for example `[1, 3, 224, 224]`. |
| `type` | Expected data type, such as `tensor(float)`. |

**Example:**

```python
for inp in session.get_inputs():
    print(f"name={inp.name}, shape={inp.shape}, type={inp.type}")
```

---

### `get_outputs()`

Returns metadata for all model outputs.

```python
outputs = session.get_outputs()
```

**Example:**

```python
for out in session.get_outputs():
    print(f"name={out.name}, shape={out.shape}, type={out.type}")
```

---

### `run()`

Executes inference with the given input data.

```python
outputs = session.run(
    output_names=None,
    input_feed={},
    run_options=None,
)
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `output_names` | `list[str]` or `None` | No | Names of outputs to return. Pass `None` to return all outputs. |
| `input_feed` | `dict` | Yes | Maps input names to input data. Keys must match the model input names. |
| `run_options` | `RunOptions` | No | Per-call execution settings. Optional for most use cases. |

**Returns:** A list containing the requested model outputs.

**Example:**

```python
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_data})
```

---

### `get_providers()`

Returns the execution providers registered for the session.

```python
providers = session.get_providers()
print(providers)
```

**Example output:**

```text
['CPUExecutionProvider']
```

Use this method when you need to inspect which execution providers are available to the session.