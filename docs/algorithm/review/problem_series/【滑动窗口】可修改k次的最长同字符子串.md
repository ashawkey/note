## [考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

> 给你一个字符串 answerKey ，其中 answerKey[i] 是第 i 个问题的正确结果。除此以外，还给你一个整数 k ，表示你能进行以下操作的最多次数：
>
> 每次操作中，将问题的正确答案改为 'T' 或者 'F' （也就是将 answerKey[i] 改为 'T' 或者 'F' ）。
> 请你返回在不超过 k 次操作的情况下，最大 连续 'T' 或者 'F' 的数目。



贪心，只需要控制滑动窗口内的相异字符数量不超过k即可。

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

