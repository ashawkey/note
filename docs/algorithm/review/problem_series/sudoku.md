### [solve sodoku](https://leetcode-cn.com/problems/sudoku-solver/)

DFS回溯即可，可以用bit操作优化`check()`，但没必要（

```cpp
class Solution {
public:
    bool check(vector<vector<char>>& board, int i, int j, char c) {
        int ii = i / 3 * 3, jj = j / 3 * 3;
        for (int k = 0; k < 9; k++) {
            if (board[i][k] == c) return false;
            if (board[k][j] == c) return false;
            if (board[ii + k % 3][jj + k / 3] == c) return false;
        }
        return true;
    }
    bool finished = false;
    void dfs(vector<vector<char>>& board, queue<pair<int,int>> q) {
        if (q.empty()) {
            finished = true;
            return;
        }
        auto [i, j] = q.front(); q.pop();
        for (char c = '1'; c <= '9'; c++) {
            if (check(board, i, j, c)) {
                board[i][j] = c;
                dfs(board, q);
                if (finished) return;
                board[i][j] = '.';
            }
        }
    }
    void solveSudoku(vector<vector<char>>& board) {
        // brute-force dfs
        queue<pair<int,int>> q;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    q.emplace(i, j);
                }
            }
        }
        finished = false;
        dfs(board, q);
        // assert finished;
    }
};
```

