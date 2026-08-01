# 后缀数组 (Suffix Array)

## 定义

按字典序排序的后缀的位置：`SA[i] = Rank(Suffix(i))`，即名次为 i 的后缀的**起始位置**。

```c++
"banana"
sa[0] = 5   a
sa[1] = 3   ana
sa[2] = 1   anana
sa[3] = 0   banana
sa[4] = 4   na
sa[5] = 2   nana
```

## 倍增法构建（$O(n\log n)$）

先按 1-后缀排序，再倍增到 2、4、… 后缀，每轮基数排序 + 重排名。**j-后缀**：

```
1-suffix: b, a, n, a, n, a
2-suffix: ba, an, na, an, na, a
4-suffix: bana, anan, nana, ana, na, a
```

```c++
const int maxn = 200005;
int wa[maxn], wb[maxn], wv[maxn], Ws[maxn];
int sa[maxn]; // sa[i] 是名次为 i 的后缀的位置

void buildSA(int* s, int* sa, int n, int m) {
    int i, j, p, *pm = wa, *k2sa = wb, *t;
    // k1 radix sort
    for (i = 0; i < m; i++) Ws[i] = 0;
    for (i = 0; i < n; i++) Ws[pm[i] = s[i]]++;
    for (i = 1; i < m; i++) Ws[i] += Ws[i - 1];
    for (i = n - 1; i >= 0; i--) sa[--Ws[pm[i]]] = i;
    // loops j -> 2j
    for (j = p = 1; p < n; j <<= 1, m = p) {
        // generate k2sa (第二关键字排序的结果)
        for (p = 0, i = n - j; i < n; i++) k2sa[p++] = i; // null k2
        for (i = 0; i < n; i++) if (sa[i] >= j) k2sa[p++] = sa[i] - j;
        // k2 radix sort
        for (i = 0; i < m; i++) Ws[i] = 0;
        for (i = 0; i < n; i++) Ws[wv[i] = pm[k2sa[i]]]++;
        for (i = 1; i < m; i++) Ws[i] += Ws[i - 1];
        for (i = n - 1; i >= 0; i--) sa[--Ws[wv[i]]] = k2sa[i];
        // update pm (重排名)
        for (t = pm, pm = k2sa, k2sa = t,
             pm[sa[0]] = 0, p = i = 1; i < n; i++) {
            int a = sa[i - 1], b = sa[i];
            if (k2sa[a] == k2sa[b] && a + j < n && b + j < n &&
                k2sa[a + j] == k2sa[b + j])
                pm[sa[i]] = p - 1; // 未发现新的 2j-后缀
            else
                pm[sa[i]] = p++;   // 发现新的 2j-后缀
        } // 当 p 达到 n，说明已有 n 个不同的 2j-后缀且排好序
    }
}
```

## height 数组与 LCP

- `Rank[i]`：位置 i 的后缀的名次，`Rank[SA[i]] = i`；
- `LCPL(i, j)`：**名次**为 i、j 的两个后缀的最长公共前缀长度；
- `height[i] = LCPL(i-1, i)`：名次 i 与 i−1 的后缀的 LCPL。

**LCP 引理**：

- 引理 1：$LCPL(i, j) = \min\{height[k] \mid k = i+1, \cdots, j\}$——即区间 height 的 RMQ（O(1) 查询用 ST 表）；
- 引理 2：$\forall i \le k < j:\ LCPL(k, j) \ge LCPL(i, j)$；
- 引理 3（H 定理）：$H[i] \ge H[i-1] - 1$，其中 $H[i] = LCPL(Rank[i]-1, Rank[i])$——据此可以 $O(n)$ 线性求 height。

```c++
int Rank[maxn], height[maxn];
void buildHeight(int* str, int n, int* sa) {
    int i, j, k;
    for (i = 0; i < n; i++) Rank[sa[i]] = i;
    height[0] = 0;
    for (i = k = 0; i < n - 1; height[Rank[i++]] = k)  // i 是位置
        for (k ? k-- : 0, j = sa[Rank[i] - 1];          // 由 H 定理，k 至少减 1
             str[i + k] == str[j + k]; k++);
}
```

## RMQ：ST 表模板

静态区间最值，预处理 $O(n\log n)$、查询 $O(1)$（注意 `arr` 不能修改）：

```c++
const int maxn = 1005;
int N;
int arr[maxn];
int st[maxn][32]; // log_2 maxn < 32

void build() {
    for (int i = 1; i <= N; i++) st[i][0] = arr[i];
    int k = log2(N * 1.0);
    for (int j = 1; j <= k; j++) {
        for (int i = 1; i <= N; i++) {
            if (i + (1 << (j - 1)) <= N) {
                st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
}

int query(int l, int r) {
    int k = log2(r - l + 1.0);
    return max(st[l][k], st[r + 1 - (1 << k)][k]);
}
```

## 应用

### [POJ 2774 Long Long Message](http://poj.org/problem?id=2774)（两个字符串的最长公共子串）

拼接两个串（中间用**未出现过的字符**隔开）→ 最长公共子串 = 属于不同串的两个相邻名次后缀的 LCP 最大值，即遍历 `height` 找最大、且 `sa[i]` 与 `sa[i-1]` 分属两个串：

```c++
int la = strlen(a), lb = strlen(b), l = 0;
for (int i = 0; i < la; i++) s[l++] = a[i] - 'a' + 1;
s[l++] = 28;  // 分隔符，未在串中出现过
for (int i = 0; i < lb; i++) s[l++] = b[i] - 'a' + 1;
s[l] = 0;
buildSA(s, sa, l + 1, 255);
buildHeight(s, l + 1, sa);
int ans = 0;
for (int i = 1; i <= l; i++) {
    if (sa[i - 1] < la && sa[i] > la || sa[i - 1] > la && sa[i] < la)
        ans = max(ans, height[i]); // 两个后缀来自不同字符串
}
```

### [POJ 3450 Corporate Identity](http://poj.org/problem?id=3450)（多字符串最长公共子串）

N 个串拼接（每个串后跟一个独特的标记字符），后缀数组 + **二分答案**：`check(m)` 检查是否存在一段连续的 height ≥ m 覆盖了所有 N 个串（用 `id[]` 标记每个后缀属于哪个串）：

```c++
bool check(int m) {
    int cnt = 0;
    memset(vis, 0, sizeof(vis));
    for (int i = 1; i <= l; i++) {
        if (height[i] < m) {           // 最长公共子串长度小于 m，失败，重置
            cnt = 0;
            memset(vis, 0, sizeof(vis));
            continue;
        }
        if (!vis[id[sa[i - 1]]]) { vis[id[sa[i - 1]]] = 1; cnt++; }
        if (!vis[id[sa[i]]])     { vis[id[sa[i]]] = 1; cnt++; }
        if (cnt == N) {
            for (int j = 0; j < m; j++) ans[j] = s[sa[i] + j] + 'a' - 1;
            ans[m] = '\0';
            return true;
        }
    }
    return false;
}
// 主流程：二分 m，check(m) 可行则 m+1，最后输出 ans
```

### [POJ 1743 Musical Theme](http://poj.org/problem?id=1743)（不重叠的最长重复子串）

相邻差做差分（避免"转调"问题），二分长度 + 分组：height ≥ m 的连续段内维护 `sa` 的 min/max，差值 ≥ m 说明存在不重叠的重复子串：

```c++
bool check(int m) {
    int mn = sa[1], mx = sa[1];
    for (int i = 2; i < N; i++) {
        if (height[i] >= m) {
            mn = min(mn, sa[i]);
            mx = max(mx, sa[i]);
        }
        else {
            if (mx - mn >= m) return true;  // 同一组内位置差 ≥ m → 不重叠
            else mx = mn = sa[i];
        }
    }
    return mx - mn >= m;
}
```

### [POJ 3261 Milk Patterns](http://poj.org/problem?id=3261)（可重叠、至少 k 次的最长重复子序列）

同样二分 + 分组：统计 height ≥ m 的连续段内后缀个数 ≥ k：

```c++
bool check(int m) {
    int cnt = 1;
    for (int i = 0; i < N; i++) {
        if (height[i] >= m) {
            cnt++;
            if (cnt == K) return true;
        }
        else cnt = 1;
    }
    return false;
}
```
