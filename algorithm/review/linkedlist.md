### loop detection 

* Method 1: hash table

  ```c++
  unordered_set<node*> visited;
  ```

* Method 2: fast-slow pointer

  `O(1)` Space.

  fast一次走两格，slow一次走一格；两指针相遇则说明有环。

  若要找到环头，则相遇之后再从起点跑一个指针，此指针与slow指针相遇时即到达环头。

  