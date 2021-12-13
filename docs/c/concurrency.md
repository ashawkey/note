## C++ Concurrency

### Concepts

* **hardware threads**: a physical CPU core.
  * hyper-threading (Intel): use one physical core but act as two virtual cores.
* **process**: the instance of a program executed by one or many threads.
  * a single process can run multiple threads on different cores.
* **(software) threads**: a light-weight unit to execute programs.
* **Multithreading**: creating multiple threads in the same process. 
* **Multiprocessing**: forking multiple processes.

We mainly talk about **multithreading** here.

