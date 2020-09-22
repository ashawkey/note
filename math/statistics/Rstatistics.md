# R statistics

### Basics

```R
x = c(1,2,3,4,5,6)
length(x)
mean(x)
median(x)
var(x)
sd(x)
cor(x, y) // correlation
cov(x, y) // covariance
```



### Distributions

```R
pnorm(x, mean, sd)
```



### Quantiles

默认计算$P[x \le z_p] = p$，即下分位数。

使用`lower.tail=False`计算上分位数。

```R
qnorm(p, mean, sd, [lower.tail=True]) # 1-z_p
qnorm(p, mean, sd, lower.tail=False) # z_p

qt(p, df)
qchisq(p, df)
qf(p, df1, df2)
```



#### Confidence Interval

居然没有原生函数（

```R
# sigma^2_x/sigma^2_y 
confint <- function(x, y, alpha){
    lower = (var(x)/var(y))/qf(alpha/2, length(x)-1, length(y)-1, lower.tail=F)
    higher = (var(x)/var(y))/qf(1-alpha/2, length(x)-1, length(y)-1, lower.tail=F)
    return(c(lower, higher))
}

# u_x - u_y
confint <- function(x, y, alpha, ratio){
    n = length(x)
    m = length(y)
    Sb2 = ((n-1)*var(x)/(ratio^2) + (m-1)*var(y))/(n+m-2)
    lower = mean(x)-mean(y) - qt(alpha/2, n+m-2, lower.tail=F)*Sb2*sqrt((ratio^2/n)+(1/m))
    higher = mean(x)-mean(y) + qt(alpha/2, n+m-2, lower.tail=F)*Sb2*sqrt((ratio^2/n)+(1/m))
    return(c(lower, higher))
}
```





### test

