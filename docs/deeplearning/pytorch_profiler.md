## profiling in pytorch

Never use `time.time()` for CUDA time measuring!

The correct way to go:

```python
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

starter.record()
# do the things
ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"{curr_time}"); starter.record()
```

