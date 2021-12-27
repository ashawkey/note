## [最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

>给你一个整数数组 arr 和一个整数 difference，请你找出并返回 arr 中最长等差子序列的长度，该子序列中相邻元素之间的差等于 difference 。

$O(n)$ 哈希即可（也可以看成是动态规划）。

```cpp
class Solution {
public:
    int longestSubsequence(vector<int>& arr, int difference) {
        unordered_map<int, int> m;
        int ans = 1;
        for (int i = 0; i < arr.size(); i++) {
            int x = arr[i];
            int y = x - difference;
            if (m.count(y)) m[x] = m[y] + 1;
            else m[x] = 1;
            //cout << x << " = " << m[x] << endl;
            ans = max(ans, m[x]);
        }
        return ans;
    }
};
```

