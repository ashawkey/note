import _adder as _backend

class Adder():
    def __init__(self):
        
        # init cuda class impl
        # TODO: this tradition is learnt from tiny-cuda-nn, but I don't know if there is any better way.
        self.impl = _backend.create_adder()

    def __call__(self, x, y):

        res = self.impl.call(x, y)

        return res
