# Linear Table

| OP             | Vector | Linked List |
| -------------- | ------ | ----------- |
| `getValue(p)`  | 1      | n           |
| `getPos(v)`    | n      | n           |
| `insert(p, v)` | n      | 1           |
| `append(v)`    | 1      | 1           |
| `delete(p)`    | n      | 1           |

### Vector (Sequential list)

Static storage space.

### Linked List

Dynamic storage space.

#### Single

* Why Head Node ?
  * 统一链表中第一个位置和其他位置上的操作。
  * 统一空表和非空表的操作。

#### Double

More space for less time.

#### Loop (single/double)

Link the head and tail.


# Exercise

* Floyd Cycle detection (& Variations)

  ```c++
  bool isLoop(node* head){
      node* first,second = head;
      while(first->next != NULL){
          first = first->next;
          if (first->next == NULL) return false;
          first = first->next;
          second = second->next;
          if (first == second) return true;
      }
  }
  
  int loopsize(node* head){
      node* first,second = head;
      int size = 0;
      int cnt = 0;
      while(first->next != NULL){
          cnt++;
          first = first->next;
          if (first->next == NULL) return -1; // no loop
          first = first->next;
          second = second->next;
          if (first == second){
              if(size==0) size=cnt;
              else return cnt-size;
          }
      }
  }
  
  node* loopenter(node* head){
      node* first,second = head;
      while(first->next != NULL){
          first = first->next;
          if (first->next == NULL) return NULL;
          first = first->next;
          second = second->next;
          if (first == second) break;  // meet point
      }
      second = head;
      while(second!=first){  
          second = second->next;
          first = first->next;
      } 
      return first;  // enter point
  }
  ```

  * 环长证明：

  Outer path = $n$ (contact point excluded), Inner path = $m$. 

  Meet at time $t$ for the first time, $t'$ for the second time:

  第二次相遇一定是f比s多走了**一圈.**

  （m奇数时均走m步，f第一圈不经过相遇点，m偶数时f走两圈，每圈m/2）
  

$$
\displaylines{

  2t - t = km \\
  2t'-t' = (k+1)m \\
  \therefore t'-t = m
  
}
$$


  * 入口点证明：

  设s入环后又走了x步，则：
  

$$
\displaylines{

  t = n + x \\
  t = km \\
  \therefore km = n +x \\
  (k-1)m + (m-x) = n \\
  
}
$$


  即，外部路径长度为当前位置走到入口点的距离加上k个环。


* Variants:

  判断两个无环单向链表是否相交，如果相交，求出交点。

  * 人为构环（A’s tail->B’s head），之后用Floyd判断入环点。
  * （更简单直白）先分别遍历两个链表，如果终点相同，则他们相交。记录他们的长度，再次遍历，先让长链表指针移动长度的差值次，再同步移动两个指针，相等处即交点。


### Others

- dancing links algorithm (Knuth)

  http://www.cnblogs.com/grenet/p/3145800.html

