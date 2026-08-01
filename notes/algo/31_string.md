# 字符串 (String Algorithms)

## KMP

### next 数组与匹配模板

```cpp
// 返回 p 在 t 中第一次出现的位置（未找到返回 -1）
int strStr(string& t, string& p) {
    int n = t.size(), m = p.size();
    if (m == 0) return 0;
    // get next
    vector<int> next(m);
    for (int i = 1, j = 0; i < m; i++) {
        while (j > 0 && p[i] != p[j]) j = next[j - 1];
        if (p[i] == p[j]) j++;
        next[i] = j;
    }
    // match
    for (int i = 0, j = 0; i < n; i++) {
        while (j > 0 && t[i] != p[j]) j = next[j - 1];
        if (t[i] == p[j]) j++;
        if (j == m) return i - m + 1;
    }
    return -1;
}
```

### 循环匹配（母串可循环）

匹配时对母串下标取模即可（`i % n`），循环上界保证不无限循环：

```cpp
int strStrCyclic(string& t, string& p) {
    int n = t.size(), m = p.size();
    if (m == 0) return 0;
    vector<int> next(m);
    for (int i = 1, j = 0; i < m; i++) {
        while (j > 0 && p[i] != p[j]) j = next[j - 1];
        if (p[i] == p[j]) j++;
        next[i] = j;
    }
    for (int i = 0, j = 0; i - j < n; i++) {
        // note the `i % n`, since we want a cyclic t.
        while (j > 0 && t[i % n] != p[j]) j = next[j - 1];
        if (t[i % n] == p[j]) j++;
        if (j == m) return i - m + 1;
    }
    return -1;
}
```

### 最小覆盖子串

**定理：字符串的最小覆盖子串长度为 `N - next[N]`**（前 N−next[N] 个字符构成的子串），且当且仅当 `N % (N - next[N]) == 0` 时它恰好由该子串重复而成。

```c++
// 判断 s 是否由某个子串重复构成
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int len = s.size();
        s += "$";
        vector<int> next(len+1, 0);
        next[0] = -1;
        int i=0, k=-1;
        while(i<len){
            while(k>=0 && s[i] != s[k]) k = next[k];
            i++, k++;
            next[i] = k;
        }
        // minimum cover substring
        int cov = len - next[len];
        return cov<len && len%cov==0;
    }
};
```

（本题还有个不讲道理的一行 trick：`(s+s).substr(1, 2*s.size()-2).find(s) != -1`。）

## 字符串哈希（Rolling Hash）

**Polynomial Rolling Hash**：

$$
hash(s) = \sum_{i=0}^{L} s[L-i-1] \cdot P^i \mod M
$$

- P 取略大于字符集大小的素数，M 取大素数（如 1e9+7）；
- 因为是"前缀和"形式，**子串哈希可以 O(1) 求**：

$$
hash(s[i..j]) = hash(s[0..j]) - hash(s[0..i-1]) \cdot P^{j-i+1}
$$

```cpp
// init
const static int P = 13; // a large prime, try-and-error
vector<long long> h(n + 1, 0), p(n + 1, 1);
for (int i = 1; i <= n; i++) {
    h[i] = h[i-1] * P + s[i-1];
    p[i] = p[i-1] * P;
}
// query hash of s[i:j+1] (0-based)
long long hash = h[j+1] - h[i] * p[j-i+1];
```

> 注意：P 的选择需要尝试（有些 P 会撞），这是哈希的固有风险；需要严格正确时用双哈希或改用后缀数组。

## Manacher（最长回文子串）

$O(n)$ 求所有回文半径。**C++ 写法比较繁琐**：

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int len = s.size();

        // build assistant string
        string ss = "^";
        for(int i=0; i<len; i++) {
            ss += "#";
            ss += s[i];
        }
        ss += "#$";

        // arm length
        int m[2005];
        int p = 0, po = 0;
        for (int i = 1; i <= 2 * len + 1; i++){
            if (p > i) m[i] = min(p-i, m[2*po-i]);
            else m[i] = 1;
            while (ss[i-m[i]] == ss[i+m[i]]) m[i]++;
            if (m[i] + i > p){
                p = m[i] + i;
                po = i;
            }
        }
        po = 1;
        for (int i = 1; i <= 2 * len + 1; i++){
            if (m[i] > m[po]) po = i;
        }

        // get the longest palindrome
        string tmp = ss.substr(po-m[po]+1, 2*m[po]-1);
        string ans;
        for(int i=0; i<tmp.size(); i++){
            if(tmp[i]!='#') ans+=tmp[i];
        }
        return ans;
    }
};
```

回文子串**数量**（Python 版更清晰）：`sum((v+1)//2 for v in manachers(S))`。

```python
class Solution(object):
    def countSubstrings(self, S):
        def manachers(S):
            A = '@#' + '#'.join(S) + '#$'
            Z = [0] * len(A)
            center = right = 0
            for i in range(1, len(A) - 1):
                if i < right:
                    Z[i] = min(right - i, Z[2 * center - i])
                while A[i + Z[i] + 1] == A[i - Z[i] - 1]:
                    Z[i] += 1
                if i + Z[i] > right:
                    center, right = i, i + Z[i]
            return Z

        return sum((v+1)//2 for v in manachers(S))
```

（回文**子序列**是区间 DP，见 `07_dynamic_programming`。）

## 例题

### [686 重复叠加字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)

`a` 重复叠加多少次后 `b` 成为其子串。三种做法，$O(n+m)$ 的两种：

**贪心 + `find`**（先叠到长度 ≥ an + bn，再判断）：

```cpp
class Solution {
public:
    int repeatedStringMatch(string a, string b) {
        int an = a.size(), bn = b.size();
        string aa;
        while (aa.size() <= bn + an) aa += a; // important!
        int index = aa.find(b);
        if (index == -1) return -1;
        if (an - index >= bn) return 1;
        return (bn + index - an - 1) / an + 2;
    }
};
```

**循环 KMP**（母串循环匹配，模板见上）：

```cpp
class Solution {
public:
    int strStr(string& t, string& p) { /* 循环 KMP 模板，见上文 */ }

    int repeatedStringMatch(string a, string b) {
        int an = a.size(), bn = b.size();
        int index = strStr(a, b);
        if (index == -1) return -1;
        if (an - index >= bn) return 1;
        return (bn + index - an - 1) / an + 2;
    }
};
```

**循环 Rabin-Karp**（滚动哈希）：

```cpp
class Solution {
public:
    constexpr int kMod1 = 1e9 + 7;
    constexpr int kMod2 = 1337;
    int strStr(string haystack, string needle) {
        int n = haystack.size(), m = needle.size();
        if (m == 0) return 0;
        long long hash_needle = 0;
        for (auto c : needle) {
            hash_needle = (hash_needle * kMod2 + c) % kMod1;
        }
        long long hash_haystack = 0, extra = 1;
        for (int i = 0; i < m - 1; i++) {
            hash_haystack = (hash_haystack * kMod2 + haystack[i % n]) % kMod1;
            extra = (extra * kMod2) % kMod1;
        }
        for (int i = m - 1; (i - m + 1) < n; i++) {
            hash_haystack = (hash_haystack * kMod2 + haystack[i % n]) % kMod1;
            if (hash_haystack == hash_needle) {
                return i - m + 1;
            }
            hash_haystack = (hash_haystack - extra * haystack[(i - m + 1) % n]) % kMod1;
            hash_haystack = (hash_haystack + kMod1) % kMod1;
        }
        return -1;
    }

    int repeatedStringMatch(string a, string b) {
        int an = a.size(), bn = b.size();
        int index = strStr(a, b);
        if (index == -1) return -1;
        if (an - index >= bn) return 1;
        return (bn + index - an - 1) / an + 2;
    }
};
```

### [187 重复的 DNA 序列](https://leetcode-cn.com/problems/repeated-dna-sequences/)

长度为 10 的子串出现超过一次。三种做法：

**滑动窗口 + unordered_map**（$O(NL)$）：

```cpp
class Solution {
    const int L = 10;
public:
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> ans;
        unordered_map<string, int> cnt;
        int n = s.length();
        for (int i = 0; i <= n - L; ++i) {
            string sub = s.substr(i, L);
            if (++cnt[sub] == 2) {
                ans.push_back(sub);
            }
        }
        return ans;
    }
};
```

**位运算优化**（$O(N)$）：4 个字母用 2 bit 表示，10 个字符 20 bit 用一个 int：

```cpp
class Solution {
    const int L = 10;
    unordered_map<char, int> bin = {{'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}};
public:
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> ans;
        int n = s.length();
        if (n <= L) return ans;
        int x = 0;
        for (int i = 0; i < L - 1; ++i) {
            x = (x << 2) | bin[s[i]];
        }
        unordered_map<int, int> cnt;
        for (int i = 0; i <= n - L; ++i) {
            x = ((x << 2) | bin[s[i + L - 1]]) & ((1 << (L * 2)) - 1);
            if (++cnt[x] == 2) {
                ans.push_back(s.substr(i, L));
            }
        }
        return ans;
    }
};
```

**Polynomial rolling hash + 前缀和**（$O(N)$，用 5 作基数，注意 P 要试）：

```cpp
class Solution {
    const static int P = 5;
    const static int M = 1e9+7;
    map<char, int> m{{'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}};
public:
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> ans;
        if (s.size() <= 10) return ans;
        vector<long long> h(s.size() + 1, 0);
        vector<long long> p(s.size() + 1, 1);
        for (int i = 1; i <= s.size(); i++) {
            h[i] = ((h[i-1] * P) % M + m[s[i-1]]) % M;
            p[i] = (p[i-1] * P) % M;
        }
        map<long long, int> cnt;
        for (int i = 1; i <= s.size() - 9; i++) {
            int j = i + 9;
            long long hash = h[j] - (h[i-1] * p[j-i+1]) % M;
            if (++cnt[hash] == 2) {
                ans.push_back(s.substr(i-1, 10));
            }
        }
        return ans;
    }
};
```

## 相关

- 最长公共子串/子序列 DP 见 `07_dynamic_programming`；多序列最长公共子串可用后缀数组（见 `33_suffix_array`）。
