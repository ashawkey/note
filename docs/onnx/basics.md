# ONNX: Open Neural Network eXchange

ONNX: An open standard format for ML models.

ONNX Runtime: a multi-platform and hardware performance-focused engine for ONNX models.


## Python API

Install:

```bash
pip install numpy protobuf==3.16.0
pip install onnx onnxruntime onnxruntime_gpu
```

Check model IO size:

```python
import onnxruntime

sess = onnxruntime.InferenceSession(path)

# single
print(sess.get_inputs()[0].name)

# loop all
for x in sess.get_inputs():
    print(x.name, x.shape, x.type)
for x in sess.get_outputs():
    print(x.name, x.shape, x.type)

# util
def inspect(path):
    import onnxruntime
    sess = onnxruntime.InferenceSession(path)
    for x in sess.get_inputs():
        print(x.name, x.shape, x.type)
    for x in sess.get_outputs():
        print(x.name, x.shape, x.type)
    
```

Update IO size:

```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load(path)

# Here both 'seq', 'batch' and -1 are dynamic using dim_param.
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'input_name': ['seq', 'batch', 3, -1]}, {'output_name': ['seq', 'batch', 1, -1]})
```

Save:

````python
onnx.save(onnx_model, path)
````

