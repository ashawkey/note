### [前n个整数中数字1的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

公式需要归纳规律，详见[here](https://leetcode-cn.com/problems/number-of-digit-one/solution/shu-zi-1-de-ge-shu-by-leetcode-solution-zopq/).

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        int ans = 0;
        // loop digit
        long long p = 1; // p = 10^k
        for (int k = 0; n >= p; k++) {
            ans += (n / (p * 10)) * p + min(max(n % (p * 10) - p + 1, 0LL), p);
            p *= 10;
        }
        return ans;
    }
};
```


### [整数序列中第N位数字](https://leetcode-cn.com/problems/nth-digit/)

在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 `n` 位数字。

仍然是归纳规律。

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

