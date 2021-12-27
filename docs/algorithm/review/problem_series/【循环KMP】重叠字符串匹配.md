## [循环字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)

给定两个字符串 a 和 b，寻找重复叠加字符串 a 的最小次数，使得字符串 b 成为叠加后的字符串 a 的子串，如果不存在则返回 -1。

注意：字符串 "abc" 重复叠加 0 次是 ""，重复叠加 1 次是 "abc"，重复叠加 2 次是 "abcabc"。



### 二分 + find

也不是不行，但仔细想想就知道这是很傻的做法了。$O(2n\log\frac{n}{m} )$。

```cpp
class Solution {
public:
    int repeatedStringMatch(string a, string b) {
        // binary search
        auto check = [&](int m) {
            string aa;
            while (m--) aa += a;
            if (aa.find(b) != -1) return true;
            else return false;
        };
        int l = 0, r = b.size() / a.size() + 1;
        while (l <= r) {
            int m = (l + r) >> 1;
            if (check(m)) r = m - 1;
            else l = m + 1;
        }
        if (check(l)) return l;
        else return -1;
    }
};
```



### find

因为其实一次find加贪心就可以得到答案：$O(2n)$

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



### 循环KMP

手写KMP并修改，使其可以循环匹配母串。$O(n+m)$

```cpp
class Solution {
public:
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
        for (int i = 0, j = 0; i - j < n; i++) { 
            // note the `i % n`, since we want a cyclic t.
            while (j > 0 && t[i % n] != p[j]) j = next[j - 1];
            if (t[i % n] == p[j]) j++;
            if (j == m) return i - m + 1;
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



### 循环Rabin-Karp

滚动哈希。$O(n+m)$

```cpp
class Solution {
public:
    constexpr int kMod1 = 1e9 + 7;
	constexpr int kMod2 = 1337;
    int strStr(string haystack, string needle) {
        int n = haystack.size(), m = needle.size();
        if (m == 0) {
            return 0;
        }
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
        if (index == -1) {
            return -1;
        }
        if (an - index >= bn) {
            return 1;
        }
        return (bn + index - an - 1) / an + 2;
    }
};
```





