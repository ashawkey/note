## NeRF gradients from instant-ngp
Basic notations:
(assume single color channel denoted as $R$)

$$
\begin{align}
\alpha_i &= 1 - e^{- \sigma_i \delta_i}
\\
T_i &= 
\begin{cases}
1, & i = 1 \\
(1 - \alpha_{i-1})(1 - \alpha_{i-2})\cdots(1 - \alpha_1), & i > 1
\end{cases} 
\\
w_i &= \alpha_i T_i
\\
R_i &= \sum_{j=1}^i w_j r_j
\\
R &= R_N
\end{align}
$$

Gradient towards color:

$$
\begin{align}
\frac {\partial L} {\partial r_i} 
&= \frac {\partial L} {\partial R} \frac {\partial R} {\partial r_i} \\
&= \frac {\partial L} {\partial R} w_i \\
&= \frac {\partial L} {\partial R} \alpha_iT_i
\end{align}
$$

Gradient towards sigma:

$$
\begin{align}
\frac {\partial L} {\partial \sigma_i} 
&= \frac {\partial L} {\partial R} \frac {\partial R} {\partial \sigma_i} \\
&= \frac {\partial L} {\partial R} \frac {\partial R} {\partial \alpha_i} \frac {\partial \alpha_i} {\partial \sigma_i} \\
\end{align}
$$

We have:

$$
\begin{align}
\frac {\partial R} {\partial \alpha_i} 
&= \frac {\partial \sum_{j=1}^N \alpha_jT_jr_j } {\partial \alpha_i} \\
&= \frac {\partial \sum_{j=i}^N \alpha_jT_jr_j } {\partial \alpha_i} \\
&= \frac {\partial (\alpha_iT_ir_i + \alpha_{i+1}T_{i+1}r_{i+1} +\cdots + \alpha_NT_Nr_N)} {\partial \alpha_i} \\
&= \frac {\partial (\alpha_iT_ir_i + \alpha_{i+1}[(1-\alpha_i)(1-\alpha_{i-1})\cdots(1-\alpha_1)] r_{i+1} +\cdots + \alpha_NT_Nr_N)} {\partial \alpha_i} \\
&\ \begin{aligned} = T_ir_i &-\alpha_{i+1}(1-\alpha_{i-1})(1-\alpha_{i-2})\cdots(1-\alpha_1)r_{i+1} \\
                            &-\cdots \\
                            &-\alpha_{N}(1-\alpha_{N-1})\cdots(1-\alpha_{i+1})(1-\alpha_{i-1})\cdots(1-\alpha_1)r_{N} \\
   \end{aligned} \\
&= \frac {1} {1-\alpha_i} (T_{i+1}r_i - \alpha_{i+1}T_{i+1}r_{i+1} -\dots -\alpha_NT_Nr_N) \\
&= \frac {1} {1-\alpha_i} (T_{i+1}r_i - \sum_{j=i+1}^N\alpha_jT_jr_j) \\
&= \frac {1} {1-\alpha_i} (T_{i+1}r_i -(R - R_i))
\end{align}
$$

And since:

$$
\frac {\partial \alpha_i} {\partial \sigma_i} = \delta_ie^{-\sigma_i\delta_i}=\delta_i(1-\alpha_i)
$$

We finally have:

$$
\begin{align}
\frac {\partial L} {\partial \sigma_i} 
&= \frac {\partial L} {\partial R} \frac {\partial R} {\partial \alpha_i} \frac {\partial \alpha_i} {\partial \sigma_i} \\
&= \frac {\partial L} {\partial R} \delta_i (1-\alpha_i) \frac {1} {1-\alpha_i} (T_{i+1}r_i -(R - R_i)) \\
&= \frac {\partial L} {\partial R} \delta_i (T_{i+1}r_i -(R - R_i))
\end{align}
$$

