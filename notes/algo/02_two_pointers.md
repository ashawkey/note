# 双指针 (Two Pointers)

## 核心思想

在**有序**（或滑动）的场景里，用两个指针代替一重循环，把 $O(n^2)$ 降到 $O(n)$。三种基本形态：

- **相向（对撞）指针**：一个在头、一个在尾，向中间靠拢——两数之和、三数之和、回文判断、救生艇。
- **同向（快慢）指针**：两个都从头部出发——链表判环、和为 s 的连续序列、滑动窗口。
- **快慢指针**：速度不同——链表环检测/环头。

关键前提：**单调性**。只有当"指针移动方向与答案方向一致"时才能保证不回溯。

## 模板

### 相向：两数之和（去重版）

```c++
// 求所有不重复的两数对，和为 target（数组先排序）
vector<vector<int>> twoSumTarget(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        int left = nums[l], right = nums[r];
        if (sum < target) {
            while (l < r && nums[l] == left) l++;   // 跳过重复
        } else if (sum > target) {
            while (l < r && nums[r] == right) r--;
        } else {
            res.push_back({left, right});
            while (l < r && nums[l] == left) l++;
            while (l < r && nums[r] == right) r--;
        }
    }
    return res;
}

// 只要一组下标：哈希 O(n) 更简单
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> v;
    for (int i = 0; i < nums.size(); i++) {
        int x = nums[i];
        if (v.count(target - x)) return vector<int>{i, v[target - x]};
        else v[x] = i;
    }
    return vector<int>{-1, -1}; // never reach here
}
```

### 同向：和为 s 的连续正整数序列

```cpp
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        int l = 1, r = 2;
        vector<vector<int>> ans;
        while (l < r && r <= (target + 1) / 2) {
            int s = (l + r) * (r - l + 1) / 2;
            if (s == target) {
                vector<int> tmp;
                for (int i = l; i <= r; i++) tmp.push_back(i);
                ans.push_back(tmp);
                l++; // don't forget this.
            }
            else if (s < target) r++;
            else l++;
        }
        return ans;
    }
};
```

### 快慢：链表判环与环头

```cpp
// 相遇后，从起点再跑一个指针，与 slow 相遇处即环头
ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }
    return nullptr;
}
```

## 例题

### [15 三数之和](https://leetcode-cn.com/problems/3sum/)

枚举一个数 a，剩下两数用相向双指针（注意跳过重复）：

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for (int first = 0; first < n; ++first) {
            if (first > 0 && nums[first] == nums[first - 1]) continue; // 去重
            int third = n - 1;
            int target = -nums[first];
            for (int second = first + 1; second < n; ++second) {
                if (second > first + 1 && nums[second] == nums[second - 1]) continue;
                while (second < third && nums[second] + nums[third] > target) --third;
                if (second == third) break;
                if (nums[second] + nums[third] == target) {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};
```

### [611 有效三角形的个数](https://leetcode-cn.com/problems/valid-triangle-number/)

排序后枚举最长边，另两条边用双指针（j、k 总共只前进 $O(n)$ 步）：

```cpp
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int ans = 0;
        for (int i = 0; i < nums.size(); i++) {
            // double pointer (j, k)
            for (int j = i + 1, k = i + 1; j < nums.size(); j++) {
                // j and k both only takes at most N ++step
                while (k + 1 < nums.size() && nums[k + 1] < nums[i] + nums[j]) k++;
                ans += max(k - j, 0);
            }
        }
        return ans;
    }
};
```

### [剑指 Offer 52 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

"换家"双指针：各自走完自己的链后换到对方链头，相遇点即交点：

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) return nullptr;
        ListNode *a = headA, *b = headB;
        while (a != b) {
            if (a) a = a->next; else a = headB;
            if (b) b = b->next; else b = headA;
        }
        return a;
    }
};
```

### [881 救生艇](https://leetcode-cn.com/problems/boats-to-save-people/)

贪心 + 双指针：每次先装最重的人，再用最轻的人填剩余空间（每艘船限载 2 人）：

```cpp
class Solution {
public:
    int numRescueBoats(vector<int>& people, int limit) {
        sort(people.begin(), people.end());
        int l = 0, r = people.size() - 1;
        int ans = 0;
        while (l <= r) {
            int rem = limit - people[r];
            if (l < r && rem >= people[l]) {
                rem -= people[l];
                l++;
            }
            ans++;
            r--;
        }
        return ans;
    }
};
```

### [825 适龄朋友](https://leetcode-cn.com/problems/friends-of-appropriate-ages/)

三个限制条件里条件三是条件二的充分条件，所以只剩两个边界。**循环里多次二分通常可优化成双指针**（两个边界都单调递增）：

```cpp
class Solution {
public:
    int numFriendRequests(vector<int>& ages) {
        int n = ages.size();
        sort(ages.begin(), ages.end());
        int left = 0, right = 0, ans = 0;
        for (int age: ages) {
            if (age < 15) continue;
            while (ages[left] <= 0.5 * age + 7) ++left;
            while (right + 1 < n && ages[right + 1] <= age) ++right;
            ans += right - left;
        }
        return ans;
    }
};
```

年龄范围只有 120，还可以用**前缀和**做到 $O(n)$（见 `04_prefix_sum`）：

```cpp
class Solution {
public:
    int numFriendRequests(vector<int>& ages) {
        vector<int> cnt(121);
        for (int age: ages) ++cnt[age];
        vector<int> pre(121);
        for (int i = 1; i <= 120; ++i) pre[i] = pre[i - 1] + cnt[i];
        int ans = 0;
        for (int i = 15; i <= 120; ++i) {
            if (cnt[i]) {
                int bound = i * 0.5 + 8;
                ans += cnt[i] * (pre[i] - pre[bound - 1] - 1);
            }
        }
        return ans;
    }
};
```

### 寻找最近数（两数之和最接近 target，Python）

```python
def solve(S, T):
    S.sort()
    i = 0
    j = len(S) - 1
    res = S[i] + S[j]
    while i < j:
        cur_val = S[i] + S[j]
        if abs(cur_val - T) < abs(res - T) or abs(cur_val - T) == abs(res - T) and cur_val < res:
            res = cur_val
        if cur_val == T:
            break
        elif cur_val < T:
            i = i + 1
        else:
            j = j - 1
    return res
```

## 相关

- 滑动窗口是同向双指针的变体，见 `03_sliding_window`；单调队列也常用于窗口极值，见 `11_stack_queue`。
