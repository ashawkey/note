# 数论与数学 (Math)

## 数论基础

### 快速幂（模意义）

```c++
// pow(n, p) % M, p >= 0
int power_modulo(int n, int p, int M) {
    int res = 1;
    while (p > 0) {
        if (p % 2 == 1) res = (res * n) % M;
        p /= 2;
        n = (n * n) % M;
    }
    return res;
}
```

### 矩阵快速幂（Fibonacci 等线性递推）

```c++
vector<vector<int>> matmul(vector<vector<int>> a, vector<vector<int>> b) {
    vector<vector<int>> res(2, vector<int>(2, 0));
    res[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    res[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    res[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    res[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
    return res;
}

vector<vector<int>> matpow(vector<vector<int>> a, int p) {
    vector<vector<int>> ans = {{1, 0}, {0, 1}}; // I
    while (p) {
        if (p % 2) ans = matmul(ans, a);
        p /= 2;
        a = matmul(a, a);
    }
    return ans;
}

// fib(n)：n >= 2 时 a = matpow({{1,1},{1,0}}, n-2); return a[0][0] + a[0][1];
```

### GCD / 素数

```c++
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

int lcm(int a, int b) {
    return gcd(a,b) > 0 ? (a / gcd(a, b)) * b : 0;
}

bool is_prime(int n) {
    if (n <= 1) return false;
    if (n == 2) return true;
    for (int i = 2; i < sqrt(n) + 1; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

// 素数筛（埃氏筛）
int is_prime[UP_LIMIT + 1];
for (int i = 1; i <= UP_LIMIT; i++) is_prime[i] = 1;
for (int i = 4; i <= UP_LIMIT; i += 2) is_prime[i] = 0;
for (int k = 3; k*k <= UP_LIMIT; k++)
    if (is_prime[k])
        for(int i = k*k; i <= UP_LIMIT; i += 2*k)
            is_prime[i] = 0;
```

### 组合数（三种规模）

```c++
using ll = long long;
ll fastpower(ll a, ll b, ll p) {
    ll res = 1;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res;
}

// 1) n <= 10^3：递推 C(n, r) = C(n-1, r-1) + C(n-1, r)

// 2) n <= 10^6：阶乘 + 乘法逆元（a^{-1} = a^{p-2} mod p, p 为素数）
const int maxn = 1e5 + 5;
const int p = 1e9 + 7;
ll fac[maxn], invfac[maxn];
void init(int n) {
    fac[0] = invfac[0] = 1;
    for (int i = 1; i < n; i++) {
        fac[i] = fac[i-1] * i % p;
        invfac[i] = invfac[i-1] * fastpower(i, p-2, p) % p;
    }
}
ll C(int a, int b) {
    if (a < b) return 0;
    return fac[a] * invfac[b] % p * invfac[a-b] % p;
}

// 3) n <= 10^18, p <= 10^5 且为素数：Lucas 定理 C(n, r) % p = C(n/p, r/p) * C(n%p, r%p) % p
ll C_large(ll a, ll b, ll p) {
    if (b > a) return 0;
    ll x = 1, y = 1;
    for (int i = 1, j = a; i <= b; i++, j--) {
        x = x * j % p;
        y = y * i % p;
    }
    return x * fastpower(y, p-2, p) % p;
}
ll lucas(ll a, ll b, ll p) {
    if (a < p && b < p) return C_large(a, b, p);
    else return C_large(a % p, b % p, p) * lucas(a / p, b / p, p) % p;
}
```

#### 组合应用：[1916 统计为蚁群构筑房间的不同顺序](https://leetcode-cn.com/problems/count-ways-to-build-rooms-in-an-ant-colony/)

树的拓扑排序数量：每个子树内相对顺序乘上"跨子树交错"的组合数：

```cpp
class Solution {
public:
    const int M = 1e9 + 7;
    using ll = long long;
    ll fac[100005], invfac[100005];
    vector<vector<int>> children;

    tuple<int,int> solve(int idx) {
        if (children[idx].empty()) return {1, 1};
        ll res = 1;
        int sum_len = 0;
        for (auto& c : children[idx]) {
            auto [len, perm] = solve(c);
            sum_len += len;
            res = res * perm % M * invfac[len] % M;
        }
        res = res * fac[sum_len] % M;
        return {sum_len + 1, res};
    }

    int waysToBuildRooms(vector<int>& prevRoom) {
        init(prevRoom.size() + 5);
        children.resize(prevRoom.size());
        for (int i = 1; i < prevRoom.size(); i++) {
            children[prevRoom[i]].push_back(i);
        }
        return get<1>(solve(0));
    }
};
```

## 递推与数学规律题

### [剑指 Offer 62 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)（约瑟夫环）

**不用模拟**，有递推公式 $f(n, m) = (f(n-1, m) + m) \bmod n$：

```cpp
class Solution {
public:
    int lastRemaining(int n, int m) {
        int ans = 0; // f(1, m) = 0
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }
};
```

#### 变体：[390 消除游戏](https://leetcode-cn.com/problems/elimination-game/)

对称性 + 递归：$f(n) = 2 \cdot (n/2 + 1 - f(n/2))$：

```cpp
class Solution {
public:
    int lastRemaining(int n) {
        return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
    }
};
```

### [458 可怜的小猪](https://leetcode-cn.com/problems/poor-pigs/)

每只猪有 `T+1` 种状态（第几轮死/不死），`buckets` 桶需要 $(T+1)^{pigs} \ge buckets$：

```cpp
class Solution {
public:
    int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
        int states = minutesToTest / minutesToDie + 1;
        int pigs = ceil(log(buckets) / log(states));
        return pigs;
    }
};
```

### [319 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

第 i 个灯泡被切换"i 的因数个数"次，亮着 ⇔ 因数个数为奇数 ⇔ i 是完全平方数：

```cpp
class Solution {
public:
    int bulbSwitch(int n) {
        return floor(sqrt(n));
    }
};
```

### [233 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

按位归纳：第 k 位（$10^k$ 位）上 1 的个数 = 高位贡献 + 低位贡献：

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        int ans = 0;
        long long p = 1; // p = 10^k
        for (int k = 0; n >= p; k++) {
            ans += (n / (p * 10)) * p + min(max(n % (p * 10) - p + 1, 0LL), p);
            p *= 10;
        }
        return ans;
    }
};
```

#### [400 第 N 位数字](https://leetcode-cn.com/problems/nth-digit/)

先定位位数 d，再定位数字，再定位位：

```cpp
class Solution {
public:
    int findNthDigit(int n) {
        int d = 1;
        while(n > d * 9 * pow(10, d - 1)) {
            n -= d * 9 * pow(10, d - 1);
            d++;
        }
        int num = pow(10, d-1) + (n - 1) / d;
        int pos = n % d;
        if(!pos) pos = d;
        int ans = int(num / pow(10, d - pos)) % 10;
        return ans;
    }
};
```

## 凸优化（无解析解，梯度下降）

### [1515 服务中心的最佳位置](https://leetcode-cn.com/problems/best-position-for-a-service-centre/)

$\min \sum_i \sqrt{(x-x_i)^2 + (y-y_i)^2}$ **没有解析解**，但目标函数是凸函数，用（mini-batch）梯度下降：

```c++
class Solution {
public:
    double getMinDistSum(vector<vector<int>>& positions) {
        double eps = 1e-7;
        double alpha = 1;
        double decay = 1e-3;
        int n = positions.size();
        int batchSize = min(n, 32);

        // mean position
        double x = 0.0, y = 0.0;
        for (const auto& pos : positions) {
            x += pos[0];
            y += pos[1];
        }
        x /= n;
        y /= n;

        mt19937 gen{random_device{}()};
        while (true) {
            shuffle(positions.begin(), positions.end(), gen);
            double xPrev = x;
            double yPrev = y;
            // mini-batch SGD
            for (int i = 0; i < n; i += batchSize) {
                int j = min(i + batchSize, n);
                double dx = 0.0, dy = 0.0;
                for (int k = i; k < j; ++k) {
                    const auto& pos = positions[k];
                    dx += (x - pos[0]) / (sqrt((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1])) + eps);
                    dy += (y - pos[1]) / (sqrt((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1])) + eps);
                }
                x -= alpha * dx;
                y -= alpha * dy;
                alpha *= (1.0 - decay); // lower lr
            }
            if (sqrt((x - xPrev) * (x - xPrev) + (y - yPrev) * (y - yPrev)) < eps) {
                break;
            }
        }
        auto getDist = [&](double xc, double yc) {
            double ans = 0;
            for (const auto& pos : positions) {
                ans += sqrt((pos[0] - xc) * (pos[0] - xc) + (pos[1] - yc) * (pos[1] - yc));
            }
            return ans;
        };
        return getDist(x, y);
    }
};
```

## 其他

- 平方根：二分或牛顿迭代（巴比伦方法），见 `01_binary_search` 与 `54_common_sense`。
- 位运算基础见 `10_bit_manipulation`。
