# 哈希 (Hash)

## 核心思想

哈希表以 $O(1)$ 均摊代价做插入/查询，把"查找值"变成"查找键"。竞赛中哈希的常见角色：

1. **计数**：统计每个值的出现次数；
2. **去重 / 查存在**：把元素放进集合；
3. **把查找变成哈希定位**：用 map 定位唯一特定值，而不是循环检验（降低一维复杂度）；
4. **前缀和 + 哈希**：子数组/子串问题中"枚举首尾 $O(n^2)$"的经典降维手段（配合状态压缩，见 `07_dynamic_programming` 的 1915）。

> C++ 中 `unordered_map` 均摊 $O(1)$，`map`（红黑树）$O(\log n)$ 但有序。需要"按 key 有序"或"找前驱后继"时才用 `map`。

## 例题

### [560 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

前缀和把"和为 K 的子数组"变成"两数之差为 K"，再用哈希计数：

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        // map of <sum, cnt>
        map<int, int> m;
        m[0] = 1;
        int sum = 0, ans = 0;
        for(int i = 0; i < nums.size(); i++){
            // cumsum
            sum += nums[i];
            // check answer
            if (m.count(sum - k)) ans += m[sum-k];
            // update map
            if (m.count(sum)) m[sum]++;
            else m[sum] = 1;
        }
        return ans;
    }
};
```

### [1218 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

$O(n)$ 哈希 DP：边扫边记"以 x 结尾的最长定差子序列长度"：

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
            ans = max(ans, m[x]);
        }
        return ans;
    }
};
```

### [剑指 Offer 35 复杂链表的复制](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

哈希 + 递归最简洁，统一处理 next 与 random 两种情况（避免重复创建）：

```cpp
class Solution {
public:
    unordered_map<Node*, Node*> m;
    Node* copyRandomList(Node* head) {
        if (head == nullptr) return nullptr;
        if (m.count(head)) return m[head];
        else {
            Node* tmp = new Node(head->val);
            m[head] = tmp;
            tmp->next = copyRandomList(head->next);
            tmp->random = copyRandomList(head->random);
            return tmp;
        }
    }
};
```

空间 $O(1)$ 的"脑内宇宙大爆炸"做法：先在每个节点后插入复制节点，再拆链：

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == nullptr) {
            return nullptr;
        }
        for (Node* node = head; node != nullptr; node = node->next->next) {
            Node* nodeNew = new Node(node->val);
            nodeNew->next = node->next;
            node->next = nodeNew;
        }
        for (Node* node = head; node != nullptr; node = node->next->next) {
            Node* nodeNew = node->next;
            nodeNew->random = (node->random != nullptr) ? node->random->next : nullptr;
        }
        Node* headNew = head->next;
        for (Node* node = head; node != nullptr; node = node->next) {
            Node* nodeNew = node->next;
            node->next = node->next->next;
            nodeNew->next = (nodeNew->next != nullptr) ? nodeNew->next->next : nullptr;
        }
        return headNew;
    }
};
```

### [432 全 O(1) 的数据结构](https://leetcode-cn.com/problems/all-oone-data-structure/)

计数并 $O(1)$ 返回最大/最小计数的键。若计数为 0 时需要删除键，只维护两个最值不够（删除后要 $O(1)$ 找到次值）——需要**双向链表维护计数顺序 + 哈希记录每个键的节点指针**，且每个链表节点放一个"同计数键集合"：

```cpp
class AllOne {
    list<pair<unordered_set<string>, int>> lst;
    unordered_map<string, list<pair<unordered_set<string>, int>>::iterator> nodes;

public:
    AllOne() {}

    void inc(string key) {
        if (nodes.count(key)) {
            auto cur = nodes[key], nxt = next(cur);
            if (nxt == lst.end() || nxt->second > cur->second + 1) {
                unordered_set<string> s({key});
                nodes[key] = lst.emplace(nxt, s, cur->second + 1);
            } else {
                nxt->first.emplace(key);
                nodes[key] = nxt;
            }
            cur->first.erase(key);
            if (cur->first.empty()) {
                lst.erase(cur);
            }
        } else { // key 不在链表中
            if (lst.empty() || lst.begin()->second > 1) {
                unordered_set<string> s({key});
                lst.emplace_front(s, 1);
            } else {
                lst.begin()->first.emplace(key);
            }
            nodes[key] = lst.begin();
        }
    }

    void dec(string key) {
        auto cur = nodes[key];
        if (cur->second == 1) { // key 仅出现一次，将其移出 nodes
            nodes.erase(key);
        } else {
            auto pre = prev(cur);
            if (cur == lst.begin() || pre->second < cur->second - 1) {
                unordered_set<string> s({key});
                nodes[key] = lst.emplace(cur, s, cur->second - 1);
            } else {
                pre->first.emplace(key);
                nodes[key] = pre;
            }
        }
        cur->first.erase(key);
        if (cur->first.empty()) {
            lst.erase(cur);
        }
    }

    string getMaxKey() {
        return lst.empty() ? "" : *lst.rbegin()->first.begin();
    }

    string getMinKey() {
        return lst.empty() ? "" : *lst.begin()->first.begin();
    }
};
```

### [2170 使数组变成交替数组的最少操作数](https://leetcode-cn.com/problems/minimum-operations-to-make-the-array-alternating/)

"计数 + 取前两大"的典型题：先把奇偶位置的计数分别放进 map，再转成 `vector<pair>` 排序取前二（map 按 key 有序，不能直接取计数前二）：

```cpp
class Solution {
public:
    int minimumOperations(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mp0, mp1;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) mp0[nums[i]]++;
           	else mp1[nums[i]]++;
        }
        vector<pair<int, int>> v0, v1;
        for (auto &[num, cnt] : mp0) v0.emplace_back(cnt, num);
        for (auto &[num, cnt] : mp1) v1.emplace_back(cnt, num);

        v0.emplace_back(0, 0); /* 存入[0,0]保证数组最少有两个元素 */
        v1.emplace_back(0, 0);

        sort(v0.begin(), v0.end(), greater<pair<int, int>>());
        sort(v1.begin(), v1.end(), greater<pair<int, int>>());

        /* 最大次数数值不等：取两个最大值相加；相等：取最大+次大 */
        if (v0[0].second != v1[0].second) return n - v0[0].first - v1[0].first;
        return n - max(v0[0].first + v1[1].first, v0[1].first + v1[0].first);
    }
};
```
