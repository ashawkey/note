# Linear Table

## Complexity of Operations

| OP             | Vector | Linked List |
| -------------- | ------ | ----------- |
| `getValue(p)`  | 1      | n           |
| `getPos(v)`    | n      | n           |
| `insert(p, v)` | n      | 1           |
| `append(v)`    | 1      | 1           |
| `delete(p)`    | n      | 1           |

## Vector (Sequential List)

静态存储空间，随机访问。

## Linked List

动态存储空间。

### Single

* 为什么带头结点（Head Node）？
  * 统一链表中第一个位置和其他位置上的操作。
  * 统一空表和非空表的操作。

### Double

More space for less time.

### Loop (single/double)

Link the head and tail.

## Exercise

### Floyd Cycle Detection (and Variations)

```c++
bool isLoop(node* head) {
    if (head == NULL) return false;
    node *first = head, *second = head;
    while (first->next != NULL) {
        first = first->next;
        if (first->next == NULL) return false;
        first = first->next;
        second = second->next;
        if (first == second) return true;
    }
    return false;
}

int loopsize(node* head) {
    node *first = head, *second = head;
    int size = 0, cnt = 0;
    while (first->next != NULL) {
        cnt++;
        first = first->next;
        if (first->next == NULL) return -1; // no loop
        first = first->next;
        second = second->next;
        if (first == second) {
            if (size == 0) size = cnt;
            else return cnt - size;
        }
    }
    return -1;
}

node* loopenter(node* head) {
    node *first = head, *second = head;
    while (first->next != NULL) {
        first = first->next;
        if (first->next == NULL) return NULL;
        first = first->next;
        second = second->next;
        if (first == second) break;  // meet point
    }
    second = head;
    while (second != first) {
        second = second->next;
        first = first->next;
    }
    return first;  // enter point
}
```

* 环长证明：

  Outer path = $n$ (contact point excluded), Inner path = $m$.

  Meet at time $t$ for the first time, $t'$ for the second time:

  第二次相遇一定是 f 比 s 多走了**一圈**（m 为奇数时均走 m 步，f 第一圈不经过相遇点；m 为偶数时 f 走两圈，每圈 m/2）。

$$
2t - t = km \\
2t' - t' = (k+1)m \\
\therefore t' - t = m
$$

* 入口点证明：

  设 s 入环后又走了 $x$ 步，则：

$$
t = n + x \\
t = km \\
\therefore km = n + x \\
(k-1)m + (m-x) = n
$$

  即，外部路径长度为当前位置走到入口点的距离加上 k 个环。

### Variants

判断两个无环单向链表是否相交，如果相交，求出交点。

* 人为构环（A's tail → B's head），之后用 Floyd 判断入环点。
* （更简单直白）先分别遍历两个链表，如果终点相同，则它们相交。记录两者的长度，再次遍历：先让长链表指针移动长度差值次，再同步移动两个指针，相等处即交点。

### Others

* Dancing Links Algorithm (Knuth)

  http://www.cnblogs.com/grenet/p/3145800.html
