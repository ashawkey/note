# [bulb switcher](https://leetcode-cn.com/problems/bulb-switcher/)

> 初始时有 n 个灯泡处于关闭状态。第一轮，你将会打开所有灯泡。接下来的第二轮，你将会每两个灯泡关闭一个。
>
> 第三轮，你每三个灯泡就切换一个灯泡的开关（即，打开变关闭，关闭变打开）。第 i 轮，你每 i 个灯泡就切换一个灯泡的开关。直到第 n 轮，你只需要切换最后一个灯泡的开关。
>
> 找出并返回 n 轮后有多少个亮着的灯泡。
>

第$i$个灯泡会被switch **$i$的因数的个数** 次。最后亮着必须有奇数个因数，只有完全平方数满足这个要求。

从而题目转换为$[1, n]$中的完全平方数个数：

```cpp
class Solution {
public:
    int bulbSwitch(int n) {
        // square root count in [1, n]
        return floor(sqrt(n));
    }
};
```

