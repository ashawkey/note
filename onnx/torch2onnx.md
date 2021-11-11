# pytorch to onnx

### export trained models to onnx format

```python
model = Network()
model.load_state_dict(pretrained_path)

# set to eval mode (must)
model.eval()

# To export the full model, we need to know IO size. 
# Usually it is fixed, but we can use dynamic axes to support random shapes (like the batch_size)
batch_size = 1
x = torch.randn(batch_size, 1, 64, 64, requires_grad=True)

output_path = "out.onnx"
torch.onnx.export(model, x, output_path, 
                  export_params=True, # export parameters, of course
                  do_constant_folding=True, # optimization
                  opset_version=11, # default is 9, some ops not supported.
                  input_names=['input'],
                  output_names=['output'],
                  # dynamic axes is a dict of {input/output name: dimension list or dict}
                  # list specifies each dynamic axis, dict supports custom name {axis: name}.
                  # use a string means it is dynamic, else a number or omitted in dict means fixed. 
                  dynamic_axes={
                      'input': {0: 'batch_size'}, # or [0], automatic named like "input_dynamic_axes_1"
                      'output': {0: 'batch_size'},
                  }
                  )

# verify the onnx model
import onnx
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)

# verify results
import onnxruntime
ort_session = onnxruntime.InferenceSession(output_path)
y = model(x) # GT
ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
np.testing.assert_allclose(y.cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
```

