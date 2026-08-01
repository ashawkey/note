# 栈 / 队列（含单调栈、单调队列）(Stack / Queue)

## 核心思想

- **栈**：后进先出，适合"最近相关"（括号匹配、逆序处理、单调栈求下一个更大/更小元素）。
- **队列**：先进先出，适合按顺序处理（BFS）。
- **单调栈 / 单调队列**：在栈/队列上维护元素值的单调性，把"暴力找极值"的 $O(n^2)$ 摊平成 $O(n)$——每个元素入/出队一次。

## 单调队列模板

保证队列内元素单调（以窗口最小值为例），通常用两个数组或 `deque` 分别记录**值**与**时间（下标）**：

```c++
//// c version (min)
int q[maxn]; // the monotone values inside the window
int p[maxn]; // the indices
int h = 0, t = 0;
for (int i = 0; i < n; i++) {
    // remove time-out
    while (h < t && p[h] <= i - k) h++;
    // maintain queue (remove vals larger than current)
    while (t >= 1 && xs[i] < q[t-1]) t--;
    p[t] = i;
    q[t++] = xs[i];
    // now q[h] is the min value in window (i-k, i]
    ...;
}

//// stl version (min)
vector<int> q; // store values
deque<int> p;  // store enqueue time
for (int i = 0; i < n; i++) {
    while (!p.empty() && p.front() <= i - k) p.pop_front();
    while (!p.empty() && xs[i] < q[p.back()]) p.pop_back(); // < or <= is both OK.
    p.push_back(i);
    q.push_back(xs[i]);
    // q[p.front()] is the min value in window (i-k, i]
    ...;
}

//// stl version (max)：把比较方向反过来
while (!p.empty() && xs[i] > q[p.back()]) p.pop_back(); // > or >= is both OK.
```

> 两个 while 各司其职：**前面**的管"过期元素出队"，**后面**的管"维护单调性"。

## 单调栈模板（下一个更大元素）

**倒序遍历** + 单调栈，弹出所有比当前小的元素后，栈顶就是"右边第一个更大的"：

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        // track next larger element for all pos in nums2
        unordered_map<int,int> hashmap;
        stack<int> st;
        for (int i = nums2.size() - 1; i >= 0; --i) {
            int num = nums2[i];
            while (!st.empty() && num >= st.top()) st.pop();
            hashmap[num] = st.empty() ? -1 : st.top();
            st.push(num);
        }
        // retrieve answer
        vector<int> res(nums1.size());
        for (int i = 0; i < nums1.size(); ++i) {
            res[i] = hashmap[nums1[i]];
        }
        return res;
    }
};
```

## 例题

### [剑指 Offer 59 滑动窗口最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

单调队列裸题，$O(N)$：

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        // monotone queue
        vector<int> q;
        deque<int> p;
        vector<int> ans;
        for (int i = 0; i < nums.size(); i++) {
            while (!p.empty() && p.front() <= i - k) p.pop_front();
            while (!p.empty() && q[p.back()] <= nums[i]) p.pop_back();
            p.push_back(i);
            q.push_back(nums[i]);
            if (i >= k - 1) ans.push_back(q[p.front()]);
        }
        return ans;
    }
};
```

另一种形式：deque 不维护时间戳，直接维护最值，配合"窗口移除时若等于队头才弹"：

```cpp
class Solution {
public:
    struct monoque{
        deque<int> que;
        void push(int n){
            while(!que.empty() && que.back()<n) que.pop_back();
            que.push_back(n);
        }
        int getmax(){ return que.front(); }
        void pop(int n){
            if(que.front() == n) que.pop_front();
        }
    };
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int N = nums.size();
        vector<int> ans;
        monoque que;
        for(int i=0; i<N; i++){
            que.push(nums[i]);
            if(i>=k-1){
                ans.push_back(que.getmax());
                que.pop(nums[i-k+1]);
            }
        }
        return ans;
    }
};
```

### [剑指 Offer 59 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

普通队列 + 单调辅助队列记录当前窗口最值。注意 `push` 里必须用 `>` 不能用 `>=`（保证同值元素在辅助队列里也一一对应）：

```cpp
class MaxQueue {
public:
    queue<int> q;
    deque<int> p;

    MaxQueue() {}

    int max_value() {
        if (q.empty()) return -1;
        return p.front();
    }

    void push_back(int value) {
        q.push(value);
        while (!p.empty() && value > p.back()) p.pop_back(); // must be >, cannot use >=
        p.push_back(value);
    }

    int pop_front() {
        if (q.empty()) return -1;
        int x = q.front(); q.pop();
        if (x == p.front()) p.pop_front();
        return x;
    }
};
```

### [918 环形子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-circular-subarray/)

**前缀和 + 单调队列**：数组复制一份变 2n，用长度为 n 的滑动窗口在前缀和上找最小值：

```c++
class Solution {
public:
    int maxSubarraySumCircular(vector<int>& A) {
        int n = A.size();
        // copy
        vector<int> AA;
        for (int i = 0; i < n; i++) AA.push_back(A[i]);
        for (int i = 0; i < n; i++) AA.push_back(A[i]);

        // cumsum
        vector<int> cs(2*n+1, 0);
        for (int i = 1; i <= 2 * n; i++) {
            cs[i] = cs[i - 1] + AA[i - 1];
        }

        vector<int> q; // store values
        deque<int> p;  // store enqueue time
        q.push_back(0);
        p.push_back(0);

        int ans = -0x3f3f3f3f;
        for (int i = 1; i <= 2 * n; i++) {
            while (!p.empty() && p.front() < i - n) p.pop_front();
            ans = max(ans, cs[i] - q[p.front()]);
            while (!p.empty() && cs[i] < q[p.back()]) p.pop_back();
            p.push_back(i);
            q.push_back(cs[i]);
        }
        return ans;
    }
};
```

（另一种思路：Kadane 两次——最大子数组，或"总和 − 最小子数组"，见 `07_dynamic_programming`。）

### [剑指 Offer 31 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

简单模拟：每次 push 后**尽可能多地 pop**：

```cpp
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> s;
        int j = 0;
        for (int i = 0; i < pushed.size(); i++) {
            s.push(pushed[i]);
            while (!s.empty() && s.top() == popped[j]) {
                s.pop();
                j++;
            }
        }
        return s.empty();
    }
};
```

### [678 有效的括号字符串](https://leetcode-cn.com/problems/valid-parenthesis-string/)

`*` 可以当 `(`、`)` 或空。仅统计数量不够（必须知道位置关系），最简单的做法是**两个栈**分别记录左括号与星号的入栈时间：

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        if (s.empty()) return true;
        stack<int> l, a;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') l.push(i);
            else if (s[i] == '*') a.push(i);
            else {
                if (!l.empty()) l.pop();
                else if (!a.empty()) a.pop();
                else return false;
            }
        }
        while (!l.empty()) {
            int t = l.top(); l.pop();
            if (a.empty()) return false;
            else {
                int ta = a.top(); a.pop();
                if (ta < t) return false;
            }
        }
        return true;
    }
};
```

更优的做法是总结规律：维护**未匹配左括号数量的可能最小值与最大值**区间：

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        if (s.empty()) return true;
        int l = 0, r = 0; // l: min possible unmatched '(', r: max possible
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') l++, r++;
            else if (s[i] == '*') l--, r++;
            else {
                l--, r--;
                if (r < 0) return false;
            };
            l = max(l, 0); // should not be neg.
        }
        return l == 0;
    }
};
```
