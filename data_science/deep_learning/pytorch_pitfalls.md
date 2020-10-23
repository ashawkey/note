# Pitfalls pytorch

## Too many open files

### Features & Error messages

* [cpu memory leakage](https://discuss.pytorch.org/t/memory-leak-on-cpu/47484)

* [DataLoader increases RAM usage every iteration](https://discuss.pytorch.org/t/dataloader-increases-ram-usage-every-iteration/6636)

* [RuntimeError: received 0 items of ancdata](https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999)

* [Too many open files](https://github.com/pytorch/pytorch/issues/11201)

  

### True Fix

when the dataset returns a dictionary containing your data, always use `deepcopy ` when you save some values elsewhere.

https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189