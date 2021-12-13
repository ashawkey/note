### [丑数](https://leetcode-cn.com/problems/ugly-number-ii/)

```cpp
// nlogn, priority_queue.
class Solution {
public:
    int nthUglyNumber(int n) {
        long long ans;
        priority_queue<long long, vector<long long>, greater<long long>> q;
        q.push(1);
        for (int i = 0; i < n; i++) {
            ans = q.top(); q.pop();
            while (!q.empty() && q.top() == ans) q.pop();
            //cout << i << " = " << ans << endl;
            q.push(ans * 2);
            q.push(ans * 3);
            q.push(ans * 5);
        }
        return ans;
    }
};


// n, pointer magic.
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> nums = {1};
        int p2 = 0, p3 = 0, p5 = 0;
        for (int i = 1; i < n; i++) {
            int x2 = nums[p2] * 2;
            int x3 = nums[p3] * 3;
            int x5 = nums[p5] * 5;
            
            int x = min(x2, min(x3, x5));
            if (x == x2) p2++;
            if (x == x3) p3++;
            if (x == x5) p5++;
            
            nums.push_back(x);
        }
        return nums.back();
    }
};
```

