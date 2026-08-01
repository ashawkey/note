# 常用技巧与模板 (Common Sense)

## 常量与常用写法

- `1e9+7`：大素数，常用作模数。
- `0x3f3f3f3f`：常用作 `+inf int`（`memset(dp, 0x3f, sizeof(dp))` 后每个字节 0x3f，整数为 0x3f3f3f3f）。
- 判 2 的幂：`(n & (n - 1)) == 0`；数 1：`while (n) { n &= n - 1; cnt++; }`（详见 `10_bit_manipulation`）。
- 平方根：二分（见 `01_binary_search`）或牛顿迭代（巴比伦法）：

```cpp
float mysqrt(float x, float eps = 1e-4) {
    float r = x, r2;
    while (true) {
        r2 = (r + x / r) / 2; // iterate to get better optimization.
        if (abs(r2 - r) < eps) break;
        else r = r2;
    }
    return r2;
}
```

- LeetCode 常见错误（AddressSanitizer）：`heap-buffer-overflow`（越界）、`stack-buffer-overflow`、`heap-use-after-free`（访问已删除数组）。

## C++ 快速 IO

```cpp
#include <iostream>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    // ...
}
```

读一整行（可能含空格）：

```cpp
#include <sstream>
string s;
getline(cin, s);       // NOT cin.getline(c_str);
istringstream iss(s);  // 从 s 中提取多个 int
int x;
while (iss >> x) { ... }
```

注意：**先 `cin >> N` 再 `getline` 会读到空行**——`cin` 不消费行尾，需要先 `getline` 读 N。

输出精度：

```cpp
cout << fixed << setprecision(5) << f << endl; // 固定小数点后 5 位
```

## Python IO 与常用操作

```python
# 读一行整数
ls = [int(x) for x in input().split()]

# 输出
print(f'{x:.2f}')                  # 保留两位小数
print(' '.join(map(str, ls)))      # 空格分隔输出
print(x, end='')                   # 不换行

# 集合删除：s.remove(x) 抛异常；s.discard(x) 不抛

# 记忆化递归
from functools import lru_cache
@lru_cache(maxsize=None)
def fib(n):
    if n < 2: return n
    return fib(n - 1) + fib(n - 2)
```

## 随机打乱（Fisher-Yates / Knuth Shuffle）

$O(N)$ 的均匀随机排列：

```cpp
void shuffle(vector<int>& nums) {
    for (int i = 0; i < nums.size(); i++) {
        int j = i + rand() % (nums.size() - i);
        swap(nums[i], nums[j]);
    }
}
```

## 摩尔投票（多数元素）

找出现次数超过 1/2 的元素，**O(1) 空间**：不同元素互相抵消。

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candi = nums[0];
        int cnt = 0;
        for (int num: nums) {
            if (candi == num) cnt++;
            else if (cnt == 0) candi = num, cnt++;
            else cnt--;
        }
        // problem asserts there always exist a majority number.
        return candi;
        // 否则需要重新数一遍验证：
        // cnt = count(candi); return cnt > n/2 ? candi : -1;
    }
};
```

超过 1/3 的元素（最多两个候选人，最后重新验证）：

```cpp
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int cnt1 = 0, cnt2 = 0;
        int candi1 = nums[0], candi2 = nums[0];
        for (int num: nums) {
            if (candi1 == num) cnt1++;
            else if (candi2 == num) cnt2++;
            else if (cnt1 == 0) candi1 = num, cnt1++;
            else if (cnt2 == 0) candi2 = num, cnt2++;
            else cnt1--, cnt2--;
        }
        int th = nums.size() / 3;
        vector<int> ans;
        cnt1 = cnt2 = 0;   // recounting
        for (int num: nums) {
            if (num == candi1) cnt1++;
            else if (num == candi2) cnt2++;
        }
        if (cnt1 > th) ans.push_back(candi1);
        if (cnt2 > th) ans.push_back(candi2);
        return ans;
    }
};
```

（栈版本：相同入栈、不同出栈，最后栈顶即多数元素。）

## 模拟题：网格旋转

[5918 循环轮转矩阵](https://leetcode-cn.com/problems/cyclically-rotating-a-grid/)：逐圈（layer）旋转，圈长 `(m + n) * 2 - 4`，旋转 k 次 = 旋转 `k % 圈长` 次，每一圈四条边依次平移：

```cpp
class Solution {
public:
    vector<vector<int>> rotateGrid(vector<vector<int>>& grid, int k) {
        int M = grid.size();
        int N = grid[0].size();
        for (int i = 0; i < min(M/2, N/2); i++) {
            int m = M - 2 * i;
            int n = N - 2 * i;
            for (int j = 0; j < (k % (m * 2 + n * 2 - 4)); j++) {
                int tmp = grid[i][i];
                for (int k = i; k < i + n - 1; k++) grid[i][k] = grid[i][k+1];
                for (int k = i; k < i + m - 1; k++) grid[k][N - i - 1] = grid[k + 1][N - i - 1];
                for (int k = N - i - 1; k > i; k--) grid[M - i - 1][k] = grid[M - i - 1][k - 1];
                for (int k = M - i - 1; k > i + 1; k--) grid[k][i] = grid[k - 1][i];
                grid[i + 1][i] = tmp;
            }
        }
        return grid;
    }
};
```

## C++ 模板头（快速解题用）

```cpp
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <climits>
#include <stack>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <cassert>
#include <unordered_map>

#define DEBUG false
#define $(x) {if (DEBUG) {cout << __LINE__ << ": "; {x} cout << endl;}}
#define _(x) {cout << #x << " = " << x << " ";}

const double E = 1e-8;
const double PI = acos(-1);
using namespace std;
```
