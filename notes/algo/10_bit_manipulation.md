# 位运算 (Bit Manipulation)

## 核心技巧速查

```c++
#define GET_BIT(n, i) (((n) & (1LL << ((i)-1))) >> ((i)-1)) // i start from 1
#define SET_BIT(n, i) ((n) | (1LL << ((i)-1)))
#define CLR_BIT(n, i) ((n) & ~(1LL << ((i)-1)))

// 2's complement: -x == (~x) + 1
// lowbit: 只保留最低位的 1
auto lowbit = [](int x) { return x & (-x); }

// highbit: 只保留最高位的 1（延迟更新技巧）
int highbit(int x) {
    int res = 0;
    for (int i = x; i != 0; i -= lowbit(i)) res = i;
    return res;
}

// 掩码取补（忽略前导 0）：00000101 -> 00000010
int maskedComplement(int x) {
    int mask = highbit(x) - 1;
    return (~x) & mask;
}

// 去掉最低位的 1
n = n & (n - 1);

// 大小写转换
('a' | ' ') == 'a';   ('A' | ' ') == 'a';
('b' & '_') == 'B';   ('B' & '_') == 'B';
('d' ^ ' ') == 'D';   ('D' ^ ' ') == 'd';
```

**应用**：判 2 的幂 `(n & (n-1)) == 0`；数 1 的个数（`while (n) { n &= n - 1; ans++; }`）；异或找只出现一次的元素。

**位运算加法模拟**（不用 +）：

```c++
int add(int a, int b) {
    while (b) {
        unsigned int c = (unsigned int)(a & b) << 1; // to handle negatives
        a ^= b;
        b = c;
    }
    return a;
}
```

## 状态压缩

用二进制位表示"集合/状态"（元素 ≤ 30 时）。典型：把单词的字母集合压成 26 位掩码；BFS 状态里带上访问集合（见 `09_bfs_dfs` 的 847 题）。

## 例题

### 数组中只出现一次的数字系列

**一个出现一次、其余两次**：全部异或。

**两个出现一次、其余两次**（[剑指 Offer 56-I](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)）：全部异或得 `a^b`，任选其中为 1 的一位做分组依据：

```cpp
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int a = 0, b = 0, i = 0;
        for (int x: nums) a ^= x;
        while ((a & (1<<i)) == 0) i++;
        a = 0;
        for (int x: nums) {
            if (x & (1<<i)) a ^= x;
            else b ^= x;
        }
        return {a, b};
    }
};
```

**一个出现一次、其余出现三次**（[剑指 Offer 56-II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)）：逐位统计出现次数模 3；通解（出现 m 次）模 m：

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums, int m = 3) { // 通解：其余数字出现 m 次
        int v[32] = {0};
        for (int x: nums) {
            for (int i = 0; i < 32; i++) {
                if (x & (1 << i)) v[i] = (v[i] + 1) % m;
            }
        }
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            if (v[i]) ans |= (1 << i);
        }
        return ans;
    }
};
```

### [869 重新排序得到 2 的幂](https://leetcode-cn.com/problems/reordered-power-of-2/)

**词频统计**：统计 n 的十进制数字词频，与某个 2 的幂（共 30 个）比对：

```cpp
class Solution {
public:
    bool check(vector<int> &num, long n) {
        vector<int> v(10);
        while(n > 0) {
            v[n % 10]++;
            n /= 10;
        }
        for(int i = 0; i < 10; i++) {
            if(v[i] != num[i]) return false;
        }
        return true;
    }

    bool reorderedPowerOf2(int n) {
        vector<int> num(10);
        while(n > 0) {
            num[n % 10]++;
            n /= 10;
        }
        for(int i = 0; i < 30; i++) {
            if(check(num, (1l << i))) return true;
        }
        return false;
    }
};
```

### [318 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)

用 26 位掩码表示单词的字母集合，O(1) 判断两词是否含公共字母：

```cpp
class Solution {
public:
    int maxProduct(vector<string>& words) {
        unordered_map<int,int> map;
        for (int i = 0; i < words.size(); i++) {
            int mask = 0;
            for (char c: words[i]) {
                mask |= 1 << (c - 'a');
            }
            if (map.count(mask)) {
                map[mask] = max(map[mask], (int)words[i].size());
            } else {
                map[mask] = words[i].size();
            }
        }
        int maxProd = 0;
        for (auto [mask1, len1] : map) {
            for (auto [mask2, len2] : map) {
                if ((mask1 & mask2) == 0) {
                    maxProd = max(maxProd, len1 * len2);
                }
            }
        }
        return maxProd;
    }
};
```

### 相关

- 数位 DP（不含连续一的非负整数，基于位与斐波那契）：`07_dynamic_programming`。
- lowbit 是树状数组的基石：`17_binary_indexed_tree`。
- 状态压缩 BFS（847 访问所有节点的最短路径）：`09_bfs_dfs`。
