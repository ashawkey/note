### [最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

计算这个字符串中最长的回文子串（Substring，连续）。

暴力以每个字符为中心向两边扩展到最长，时间复杂度$O(n^2)$。

Manacher算法：时间复杂度$O(n)$：

```cpp
// manacher in c++ is complicated...
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


### [回文子串数量](https://leetcode-cn.com/problems/palindromic-substrings/)

计算字符串中有多少个回文子串。

Manacher算法$O(n)$:

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


### [最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

计算这个字符串中最长的回文子序列的长度（Subsequence，可以不连续）。

动态规划，$O(n^2)$.

```cpp
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int len = s.size();
        if (len == 0) return 0;

        vector<vector<int>> dp(len, vector<int>(len, 0));
        
        for (int l=1; l<=len; l++){
            for (int i=0; i<=len-l; i++){
                int j = i+l-1;
                if (j==i) dp[i][j] = 1;
                else if (j==i+1) {
                    if (s[i]==s[j]) dp[i][j] = 2;
                    else dp[i][j] = 1;
                }
                else {
                    if (s[i]==s[j]) dp[i][j] = dp[i+1][j-1]+2;
                    else dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        
        return dp[0][len-1];
    }
};
```


### [不同回文子序列的数量](https://leetcode-cn.com/problems/count-different-palindromic-subsequences/)

