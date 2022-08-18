# threading and multiprocessing



### difference between threading and multiprocessing

* threading is limited by the GIL (for Cpython), so only one python process can run at the same time.



### multiprocessing

thread start method

* 'spawn': default on windows and macos, safe, slower.
* 'fork': only available and default on unix, unsafe, faster.

```python
import multiprocessing as mp

mp.set_start_method('spawn')
```





simple data-parallel:

```python
from multiprocessing import Pool

def f(x):
	do_some_thing(x)

# method 1
p = Pool(8)
p.map(f, list(range(1024)))
p.close()
p.join()

# method 2
with Pool(8) as p:
    p.map(f, list(range(1024)))
```

with return values:

```python
def f(x): 
    return x

with Pool(8) as p:
	res = p.map(f, list(range(1024)))
# res: [0, 1, 2, ..., 1023]
```



sender-receiver model with Queue:

```python
import time
from multiprocessing import Queue, Process

# sender (background)
def sender(q):
    x = 0
    while True:
        q.put(x)
        x += 1
        time.sleep(0.5)

q = Queue()
p = Process(target=sender, args=(q,))

p.start()

# reciever (foreground)
for _ in range(10):
    x = q.get()
    print(x)
    time.sleep(0.5)

p.terminate() # terminate sender (don't use join since it runs endlessly)
```

A classy way:

```python
import time
from multiprocessing import Queue, Process

class A:
    def __init__(self):
        self.q = Queue()

        # sender (background)
        def sender():
            x = 0
            while True:
                self.q.put(x)
                x += 1
                time.sleep(0.5)

        self.p = Process(target=sender)

        self.p.start()

    def run(self):

        # reciever (foreground)
        for _ in range(10):
            x = self.q.get()
            print(x)
            time.sleep(0.5)

        self.p.terminate()

a = A()
a.run()
```





### threading



sender-receiver model

```python
import time
from queue import Queue
from threading import Thread, Event

class A:
    def __init__(self):
        self.q = Queue()
        self.exit_event = Event()

        # sender (background)
        def sender():
            x = 0
            while True:
                # threading does not have a terminate() or kill(), so we manually handle the exit
                if self.exit_event.is_set():
                    break
                self.q.put(x)
                x += 1
                time.sleep(0.5)

        self.p = Thread(target=sender)

        self.p.start()

    def run(self):

        # reciever (foreground)
        for _ in range(10):
            x = self.q.get()
            print(x)
            time.sleep(0.5)
		
        # set exit event
        self.exit_event.set()
        self.p.join()


a = A()
a.run()
```



