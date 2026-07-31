##### [array rotation in-place](https://leetcode-cn.com/problems/cyclically-rotating-a-grid/)

Not difficult, just some tedious index calculation.

```python
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
                for (int k = i; k < i + n - 1; k++) {
                    grid[i][k] = grid[i][k+1];
                }
                for (int k = i; k < i + m - 1; k++) {
                    grid[k][N - i - 1] = grid[k + 1][N - i - 1];
                }
                for (int k = N - i - 1; k > i; k--) {
                    grid[M - i - 1][k] = grid[M - i - 1][k - 1];
                }
                for (int k = M - i - 1; k > i + 1; k--) {
                    grid[k][i] = grid[k - 1][i];
                }
                grid[i + 1][i] = tmp;
            }
        }
        return grid;
    }
};
```

