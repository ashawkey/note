### [有效的括号字符串](https://leetcode-cn.com/problems/valid-parenthesis-string/)

> 给定一个只包含三种字符的字符串：（ ，） 和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：
>
> * 任何左括号 ( 必须有相应的右括号 )。
>
> * 任何右括号 ) 必须有相应的左括号 ( 。
> * 左括号 ( 必须在对应的右括号之前 )。
>
> * *可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
> * 一个空字符串也被视为有效字符串。

仅仅统计左括号与星号的数量是不够的，因为必须记录左括号与星号的位置关系。

最简单的方法是用**两个栈**分别记录左括号与星号的入栈时间。

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        if (s.empty()) return true;
        stack<int> l, a;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') l.push(i);
            else if (s[i] == '*') a.push(i);
            else {
                if (!l.empty()) l.pop();
                else if (!a.empty()) a.pop();
                else return false;
            }
        }
        while (!l.empty()) {
            int t = l.top(); l.pop();
            if (a.empty()) return false;
            else {
                int ta = a.top(); a.pop();
                if (ta < t) return false;
            }
        }
        return true;
    }
};
```

更好的做法是总结规律：

维护**未匹配的左括号数量可能的最小值和最大值**。

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        if (s.empty()) return true;
        int l = 0, r = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') l++, r++;
            else if (s[i] == '*') l--, r++;
            else {
                l--, r--;
                if (r < 0) return false;
            };
            l = max(l, 0); // should not be neg.
        }
        return l == 0;
    }
};
```

