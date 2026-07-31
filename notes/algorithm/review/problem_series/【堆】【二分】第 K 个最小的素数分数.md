### [第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

给你一个按递增顺序排序的数组 arr 和一个整数 k 。数组 arr 由 1 和若干 素数  组成，且其中所有整数互不相同。

对于每对满足 0 < i < j < arr.length 的 i 和 j ，可以得到分数 arr[i] / arr[j] 。

那么第 k 个最小的分数是多少呢?  以长度为 2 的整数数组返回你的答案, 这里 answer[0] == arr[i] 且 answer[1] == arr[j] 。


### 暴力

扫描k次，手动管理大小关系只能做到每次$O(N)$。

$O(kN)=O(N^3)$，TLE。


### 堆

类似的想法，扫描k次，但通过堆管理排序。本质是**多路归并**问题。

$O(k\log N)$。

```cpp

class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        auto cmp = [&](const pair<int, int>& x, const pair<int, int>& y) {
            return arr[x.first] * arr[y.second] > arr[x.second] * arr[y.first];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
        for (int j = 1; j < n; ++j) {
            q.emplace(0, j);
        }
        for (int _ = 1; _ < k; ++_) {
            auto [i, j] = q.top();
            q.pop();
            if (i + 1 < j) {
                q.emplace(i + 1, j);
            }
        }
        return {arr[q.top().first], arr[q.top().second]};
    }
};
```


### 二分

小于$x$的分数数量关于$x$是单调递增的，所以可以二分$x$，计数小于$x$的分数数量来求解。

$O(N\log C)$, $C = \max_{v\in arr}v$。

```cpp
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        double left = 0.0, right = 1.0;
        while (true) {
            double mid = (left + right) / 2;
            
            int i = -1, count = 0;
            int x = 0, y = 1;
            
            // count how many elements are smaller than mid.
            for (int j = 1; j < n; ++j) {
                while ((double)arr[i + 1] / arr[j] < mid) {
                    ++i;
                    if (arr[i] * y > arr[j] * x) {
                        x = arr[i];
                        y = arr[j];
                    }
                }
                count += i + 1;
            }

            if (count == k) {
                return {x, y};
            }
            if (count < k) {
                left = mid;
            }
            else {
                right = mid;
            }
        }
    }
};
```


### 类似题目：[查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。

定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。

请找到和最小的 k 个数对 (u1,v1),  (u2,v2)  ...  (uk,vk) 。


#### 堆

类似的，虽然两组数可复用时没有明显的单调性，但如果固定其中一个数，求和对另一个数一定是单调的，所以也可以用堆解决。

```cpp
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        auto cmp = [&](const pair<int, int>&x, const pair<int, int>&y) {
            return nums1[x.first] + nums2[x.second] > nums1[y.first] + nums2[y.second];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
        for (int i = 0; i < nums2.size(); i++) {
            q.emplace(0, i);
        }
        vector<vector<int>> ans;
        for (int i = 0; i < k; i++) {
            if (q.empty()) break; // when k > nums1.size() * nums2.size()
            auto [m, n] = q.top(); q.pop();
            ans.push_back({nums1[m], nums2[n]});
            if (m + 1 < nums1.size()) {
                q.emplace(m + 1, n);
            }
        }
        return ans;
    }
};
```

