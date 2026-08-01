# 滑动窗口 (Sliding Window)

## 核心思想

滑动窗口是**同向双指针**的变体：维护一个 `[l, r]` 区间，随遍历扩展 r、按条件收缩 l，保证窗口内的信息始终满足某个约束。适用于**连续子数组/子串**问题。

- **定长窗口**：长度固定，r 每步 +1、l 同步 +1。
- **变长窗口**：先扩展 r 直到不满足约束，再收缩 l 直到重新满足；答案在每次合法状态时更新。
- **可修改 k 次**类问题：把"修改次数"计入约束（窗口内异类字符数 ≤ k）。

时间复杂度 $O(n)$：每个元素至多进/出窗口一次。

## 模板（变长窗口）

```cpp
int slidingWindow(vector<int>& nums, int k) {
    int l = 0, ans = 0, bad = 0; // bad = 窗口内违反约束的度量
    for (int r = 0; r < nums.size(); r++) {
        // 扩展：把 nums[r] 计入
        if (needChange(nums[r])) bad++;
        // 收缩：不满足约束时移动 l，同时从窗口中移除 nums[l]
        while (bad > k) {
            if (needChange(nums[l])) bad--;
            l++;
        }
        // 现在 [l, r] 合法，更新答案
        ans = max(ans, r - l + 1);
    }
    return ans;
}
```

## 例题

### [2024 考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

最多修改 k 个字符，求最长的全 T 或全 F 连续段。分别对 'T' 和 'F' 跑一次：窗口内**相异字符数量 ≤ k** 即可。用队列记录需要修改的位置，超出 k 时收缩：

```cpp
class Solution {
public:
    int maxConsecutiveAnswers(string answerKey, int k) {
        auto solve = [&] (char c) {
            int l = 0, r = 0, cnt = 0, ans = 0;
            queue<int> pos;
            while (r < answerKey.size()) {
                if (answerKey[r] != c) {
                    pos.push(r);
                    cnt++;
                    if (cnt > k) {
                        l = pos.front() + 1;
                        pos.pop();
                        cnt--;
                    }
                }
                ans = max(ans, r - l + 1);
                r++;
            }
            return ans;
        };
        return max(solve('F'), solve('T'));
    }
};
```

（另一种写法：不用队列，收缩时直接 `while (bad > k)` 移动 l——见模板。）

## 相关

- 窗口**最值**（不是满足约束，而是求窗口极值）用单调队列：`11_stack_queue`；
- 一般同向双指针问题：`02_two_pointers`；
- 滑窗 + 哈希/位运算做子串匹配：`31_string`（重复 DNA 序列）、`13_hash`。
