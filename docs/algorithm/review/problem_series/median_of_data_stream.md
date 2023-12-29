### [get the median of a data stream](https://leetcode-cn.com/problems/find-median-from-data-stream/)

最简单的做法是插入排序，但每次插入耗时$O(N)$。下面的两种方法均为$O(\log N)$的插入耗时。（查询耗时均为$O(1)$）

#### 双堆

很巧妙的做法，插入时可能要进行堆间的元素交换。

```cpp
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int> mx;
    priority_queue<int, vector<int>, greater<int>> mn;

    MedianFinder() {
    }
    
    void addNum(int num) {
        // logn
        if (mx.size() == mn.size()) {
            if (!mn.empty() && num > mn.top()) {
                mx.push(mn.top()); mn.pop();
                mn.push(num);
            } else {
                mx.push(num);
            }
        } else {
            if (num < mx.top()) {
                mn.push(mx.top()); mx.pop();
                mx.push(num);
            } else {
                mn.push(num);
            }
        }
    }
    
    double findMedian() {
        if (mn.size() == mx.size()) {
            return (mn.top() + mx.top()) * 0.5;
        } else {
            return mx.top();
        }
    }
};
```


#### multiset维护中点迭代器

`multiset`本身就是红黑树，具有$O(\log N)$的插入耗时，关键在于如何实现$O(1)$的查询耗时。

```cpp
class MedianFinder {
    multiset<int> data;
    multiset<int>::iterator mid;

public:
    MedianFinder() : mid(data.end()) {}

    void addNum(int num)
    {
        int n = data.size();
        data.insert(num);

        if (!n)                                 // first element inserted
            mid = data.begin();
        else if (num < *mid)                    // median is decreased
            mid = (n & 1 ? mid : prev(mid));
        else                                    // median is increased
            mid = (n & 1 ? next(mid) : mid);
    }

    double findMedian()
    {
         int n = data.size();
        return (*mid + *next(mid, n % 2 - 1)) * 0.5;
    }
};
```


