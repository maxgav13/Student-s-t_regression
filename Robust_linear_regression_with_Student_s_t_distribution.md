---
title: "Robust Linear Regression with Student's T Distribution"
author: "A Solomon Kurz"
date: "2018-03-25"
output:
  html_document:
    code_folding: show
    keep_md: TRUE
---



The purpose of this document is to demonstrate the advantages of the Student's t distribution for regression with outliers, particularly within a [Bayesian framework](https://www.youtube.com/channel/UCNJK6_DZvcMqNSzQdEkzvzA/playlists).

In this document, I’m presuming you are familiar with linear regression, familiar with the basic differences between frequentist and Bayesian approaches to fitting regression models, and have a sense that the issue of outlier values is a pickle worth contending with. All code in is [R](https://www.r-bloggers.com/why-use-r-five-reasons/), with a heavy use of the [tidyverse](http://style.tidyverse.org) and the [brms](https://cran.r-project.org/web/packages/brms/index.html) package.

Here's the deal: The Gaussian likelihood, which is typical for regression models, is sensitive to outliers. The normal distribution is a special case of Student's t distribution with $\nu$ (nu, the degree of freedom parameter) set to infinity. However, when $\nu$ is small, Student's t distribution is more robust to multivariate outliers. For more on the topic, see [Gelman & Hill (2007, chapter 6)](http://www.stat.columbia.edu/~gelman/arm/) or [Kruschke (2014, chapter 16)](https://sites.google.com/site/doingbayesiandataanalysis/).

In this document, we demonstrate how vunlerable the Gaussian likelihood is to outliers and then compare it too different ways of using Student's t likelihood for the same data.

First, we'll get a sense of the distributions with a plot.


```r
library(tidyverse)
library(viridis)

ggplot(data = tibble(x = seq(from = -6, to = 6, by = .01)), aes(x = x)) +
  geom_line(aes(y = dnorm(x))) +
  geom_line(aes(y = dt(x, df = 10)),  # note how we've defined df by nu (i.e., df)
            color = viridis_pal(direction = 1, option = "C")(6)[2]) +
  geom_line(aes(y = dt(x, df = 5)),   # note how we've defined df by nu 
            color = viridis_pal(direction = 1, option = "C")(6)[3]) +
  geom_line(aes(y = dt(x, df = 2.5)), # note how we've defined df by nu 
            color = viridis_pal(direction = 1, option = "C")(6)[4]) +
  geom_line(aes(y = dt(x, df = 1)),   # note how we've defined df by nu 
            color = viridis_pal(direction = 1, option = "C")(6)[5]) +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = -5:5) +
  labs(subtitle = "Gauss is in black. Student's t with nus of 10, 5, 2.5, and 1 range from purple to\norange.",
       x = NULL) +
  theme(panel.grid = element_blank())
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

[Asside: In this document, we make use of the handy [viridis package](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html) for our color paletes.]

So the difference is that a Student's t with a low $\nu$ will have notably heavier tails than the conventional Gaussian distribution. This difference is most notable as $\nu$ approaches 1. However, the difference can be subtle when looking at a plot. Another way is to compare how probable relatively extreme values are in a Student's t distribution relative to the Gaussian. For the sake of demonstration, here we'll compare Gauss with Student's t with a $\nu$ of 5. In the plot above, they are clearly different, but not shockingly so. However, that difference is very notable in the tails.

In the table, below, we compare the probability of a given Z score or lower within the Gaussian and a $\nu$ = 5 Student's t. In the rightmost column, we compare the probabilities in a ratio.


```r
# Here we pic our nu
nu <- 5

tibble(Z_score = 0:-5,
       p_Gauss = pnorm(Z_score, mean = 0, sd = 1),
       p_Student_t = pt(Z_score, df = nu),
       `Student/Gauss ratio` = p_Student_t/p_Gauss) %>%
  mutate_if(is.double, round, digits = 5)
```

```
## # A tibble: 6 x 4
##   Z_score   p_Gauss p_Student_t `Student/Gauss ratio`
##     <int>     <dbl>       <dbl>                 <dbl>
## 1       0 0.500         0.500                    1.00
## 2      -1 0.159         0.182                    1.14
## 3      -2 0.0228        0.0510                   2.24
## 4      -3 0.00135       0.0150                  11.1 
## 5      -4 0.0000300     0.00516                163   
## 6      -5 0             0.00205               7160
```

Note how extreme scores are more probable in this Student’s t than in the Gaussian. A consequence of this is that extreme scores are less influential to your solutions when you use a small-$\nu$ Student’s t distribution in place of the Gaussian. That is, the small-$\nu$ Student’s t is more robust than the Gaussian to unusual and otherwise influential observations.

In order to demonstrate, let's simulate our own. We'll start by creating multivariate normal data.

# Let's create our initial tibble of well-behaved data, `d`

First, we'll need to define our variance/covariance matrix.


```r
# ?matrix
s <- matrix(c(1, .6, 
              .6, 1), 
             nrow = 2, ncol = 2)
```

By the two .6s on the off-diagonal positions, we indicate we'd like our two variables to have a correlation of .6.

Second, our variables also need means, which we'll define with a mean vector.


```r
m <- c(0, 0)
```

Third, we'll use the `mvrnorm()` function from the [MASS package](https://cran.r-project.org/web/packages/MASS/index.html) to simulate our data.


```r
library(MASS)

set.seed(3)
d <- mvrnorm(n = 100, mu = m, Sigma = s)
d <- 
  d %>%
  as_tibble() %>%
  rename(y = V1, x = V2)

head(d)
```

```
## # A tibble: 6 x 2
##         y      x
##     <dbl>  <dbl>
## 1 -1.14   -0.584
## 2 -0.0805 -0.443
## 3 -0.239   0.702
## 4 -1.30   -0.761
## 5 -0.280   0.630
## 6 -0.245   0.299
```

Side note. For more information on simulating data, check out [this nice r-bloggers post](https://www.r-bloggers.com/creating-sample-datasets-exercises/).

This line reorders our tibble by `x`.


```r
d <-
  d %>%
  arrange(x)

head(d)
```

```
## # A tibble: 6 x 2
##        y     x
##    <dbl> <dbl>
## 1 -2.21  -1.84
## 2 -1.27  -1.71
## 3 -0.168 -1.60
## 4 -0.292 -1.46
## 5 -0.785 -1.40
## 6 -0.157 -1.37
```

# Let's create our outlier tibble, `o`

Here we'll make two outlying and unduly influential values.


```r
o <- d

o[c(1:2), 1] <- c(5, 4)

head(o)
```

```
## # A tibble: 6 x 2
##        y     x
##    <dbl> <dbl>
## 1  5.00  -1.84
## 2  4.00  -1.71
## 3 -0.168 -1.60
## 4 -0.292 -1.46
## 5 -0.785 -1.40
## 6 -0.157 -1.37
```

With the code, above, we replaced the first two values of our first variable, which were both quite negative, with two large positive values.

# Frequentist OLS Models

To get a quick sense of what we've done, we'll first fit two models with OLS regression. The first model, `fit0`, is of the multivariate normal data, `d`. The second model, `fit1`, is on the otherwise identical data with the two odd and influential values, `o`.


```r
fit0 <- lm(data = d, y ~ 1 + x)
fit1 <- lm(data = o, y ~ 1 + x)
```

We'll use the [broom package](https://cran.r-project.org/web/packages/broom/index.html) to assist with model summaries and other things.


```r
library(broom)

tidy(fit0) %>% mutate_if(is.double, round, digits = 2)
```

```
##          term estimate std.error statistic p.value
## 1 (Intercept)    -0.01      0.09     -0.08    0.94
## 2           x     0.45      0.10      4.55    0.00
```

```r
tidy(fit1) %>% mutate_if(is.double, round, digits = 2)
```

```
##          term estimate std.error statistic p.value
## 1 (Intercept)     0.12      0.11      1.12    0.26
## 2           x     0.15      0.13      1.21    0.23
```

Just two odd and influential values dramatically changed the model parameters, particularly the slope. Let's plot the data to get a sense of what's going on.


```r
# The well-behaived data
ggplot(data = d, aes(x = x, y = y)) +
  stat_smooth(method = "lm", color = "grey92", fill = "grey67", alpha = 1, fullrange = T) +
  geom_point(size = 1, alpha = 3/4) +
  scale_x_continuous(limits = c(-4, 4)) +
  coord_cartesian(xlim = -3:3, 
                  ylim = -3:5) +
  labs(title = "No Outliers") +
  theme(panel.grid = element_blank())
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

```r
# The data with two outliers
ggplot(data = o, aes(x = x, y = y, color = y > 3)) +
  stat_smooth(method = "lm", color = "grey92", fill = "grey67", alpha = 1, fullrange = T) +
  geom_point(size = 1, alpha = 3/4) +
  scale_color_manual(values = c("black", viridis_pal(direction = 1, option = "C")(7)[4])) +
  scale_x_continuous(limits = c(-4, 4)) +
  coord_cartesian(xlim = -3:3, 
                  ylim = -3:5) +
  labs(title = "Two Outliers") +
  theme(panel.grid = element_blank(),
        legend.position = "none")
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-11-2.png)<!-- -->

The two outliers were quite influential on the slope. It went from a nice clear diagonal to almost horizontal. You'll also note how the 95% intervals (i.e., the bowtie shape) were a bit wider when based on the `o` data.

One of the popular ways to quantify outlier status is with Mahalanobis' distance. However, the Mahalanobis distance is primarilly valid for multivariate normal data. Though the data in this example are indeed multivariate normal--or at least they were before we injected two outlying values into them--I am going to resist relying on Mahalanobis' distance. There are other more general approaches that will be of greater use when you need to explore other variants of the generalized linear model. The `broom::augment()` function will give us access to one.


```r
aug0 <- augment(fit0)
aug1 <- augment(fit1)

glimpse(aug1)
```

```
## Observations: 100
## Variables: 9
## $ y          <dbl> 5.00000000, 4.00000000, -0.16783167, -0.29164105, -...
## $ x          <dbl> -1.8439208, -1.7071418, -1.5996509, -1.4601550, -1....
## $ .fitted    <dbl> -0.155937416, -0.135213012, -0.118926273, -0.097790...
## $ .se.fit    <dbl> 0.2581834, 0.2427649, 0.2308204, 0.2155907, 0.20864...
## $ .resid     <dbl> 5.15593742, 4.13521301, -0.04890540, -0.19385084, -...
## $ .hat       <dbl> 0.05521164, 0.04881414, 0.04412882, 0.03849763, 0.0...
## $ .sigma     <dbl> 0.964211, 1.017075, 1.104423, 1.104253, 1.102081, 1...
## $ .cooksd    <dbl> 6.809587e-01, 3.820802e-01, 4.783890e-05, 6.480561e...
## $ .std.resid <dbl> 4.82755612, 3.85879897, -0.04552439, -0.17992001, -...
```

Here we can compare the observations with Cook's distance, $D_{i}$,  (i.e., `.cooksd`). $D_{i}$ is a measure of the influence of a given observation on the model. To compute $D_{i}$, the model is fit once for each $n$ case, with that case dropped. Then the difference in the model with all observations and the model with all observations but the $i$th observation, as defined by the Euclidian distance between the estimators. [Fahrmeir et al (2013, p. 166)](http://www.springer.com/us/book/9783642343322#aboutBook) suggest that within the OLS framework "as a rule of thumb, observations with $D_{i}$ > 0.5 are worthy of attention, and observations with $D_{i}$ > 1 should always be examined." Here we plot $D_{i}$ against our observation index, $i$, for both models.


```r
aug0 %>%  # The well-behaived data
  mutate(i = 1:n()) %>%
  bind_rows(  # The data with two outliers
    aug1 %>%
      mutate(i = 1:n())
    ) %>%
  mutate(fit = rep(c("fit b0", "fit b1"), each = n()/2)) %>%

  ggplot(aes(x = i, y = .cooksd)) +
  geom_hline(yintercept = .5, color = "white") +
  geom_point(alpha = .5) +
  coord_cartesian(ylim = c(0, .7)) +
  theme(panel.grid = element_blank(),
        axis.title.x = element_text(face = "italic", family = "Times")) +
    facet_wrap(~fit)
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

For the model of the well-behaved data, `fit0`, we have $D_{i}$ values all hovering near zero. However, the plot for `fit1` shows one $D_{i}$ value well above the 0.5 level and another not quite that high but deviant relative to the rest. Our two outlier values look quite influential for the results of `fit1`.

# Switching to a Bayesian Framework

In this document, we'll use the [brms package](https://cran.r-project.org/web/packages/brms/index.html) to fit our Bayesian regression models. Throughout, we'll use [weakly-regularizing priors](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).


```r
library(brms)
```

## Sticking with Gauss

For our first two Bayesian models, `b0` and `b1`, we'll use the conventional Gaussian likelihood (i.e., `family = gaussian` in the `brm()` function). Like with `fit01`, above, the first model is based on the nice `d` data. The second, `b1`, is based on the more-difficult `o` data.


```r
b0 <- 
  brm(data = d, family = gaussian,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))

b1 <- 
  brm(data = o, family = gaussian,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

Here are the model summaries.


```r
tidy(b0) %>% slice(1:3) %>% mutate_if(is.double, round, digits = 2)
```

```
## # A tibble: 3 x 5
##   term        estimate std.error  lower upper
##   <chr>          <dbl>     <dbl>  <dbl> <dbl>
## 1 b_Intercept  -0.0100    0.0900 -0.150 0.130
## 2 b_x           0.440     0.100   0.280 0.610
## 3 sigma         0.860     0.0600  0.770 0.970
```

```r
tidy(b1) %>% slice(1:3) %>% mutate_if(is.double, round, digits = 2)
```

```
## # A tibble: 3 x 5
##   term        estimate std.error   lower upper
##   <chr>          <dbl>     <dbl>   <dbl> <dbl>
## 1 b_Intercept    0.120    0.110  -0.0500 0.300
## 2 b_x            0.150    0.130  -0.0600 0.370
## 3 sigma          1.11     0.0800  0.980  1.24
```

These should look familiar. They're very much like the results from the OLS models. Hopefully this isn't surprising. Our priors were quite weak, so there's no reason to suspect the results would differ much.

### The LOO and other goodies help with diagnostics.

With the `loo()` function, we'll extract loo objects, which contain some handy output. We'll use `str()` to get a sense of what's all in there.


```r
loo_b0 <- loo(b0)
loo_b1 <- loo(b1)
```

```
## Warning: Found 1 observations with a pareto_k > 0.7 in model 'b1'. It is
## recommended to set 'reloo = TRUE' in order to calculate the ELPD without
## the assumption that these observations are negligible. This will refit
## the model 1 times to compute the ELPDs for the problematic observations
## directly.
```

```r
str(loo_b1)
```

```
## List of 9
##  $ elpd_loo   : num -156
##  $ p_loo      : num 7.02
##  $ looic      : num 312
##  $ se_elpd_loo: num 15.7
##  $ se_p_loo   : num 4.19
##  $ se_looic   : num 31.4
##  $ pointwise  : num [1:100, 1:3] -14.45 -9.12 -1.04 -1.06 -1.25 ...
##   ..- attr(*, "dimnames")=List of 2
##   .. ..$ : NULL
##   .. ..$ : chr [1:3] "elpd_loo" "p_loo" "looic"
##  $ pareto_k   : num [1:100] 0.994483 0.622069 -0.014279 -0.000441 0.002575 ...
##  $ model_name : chr "b1"
##  - attr(*, "log_lik_dim")= int [1:2] 4000 100
##  - attr(*, "class")= chr [1:2] "ic" "loo"
```

For a detailed explanation of all those elements, see the [reference manual](https://cran.r-project.org/web/packages/loo/loo.pdf). For our purposes, we'll focus on `pareto_k`. Here's what it looks like for the `b1` model.


```r
loo_b1$pareto_k
```

```
##   [1]  0.9944829854  0.6220688251 -0.0142787825 -0.0004414504  0.0025748863
##   [6] -0.0362324125 -0.1092558653 -0.0494050935  0.0945824167 -0.0172846292
##  [11]  0.0019559806 -0.0491486519 -0.0724288462 -0.0092298294 -0.0601263795
##  [16]  0.0110444224 -0.0479315912 -0.0697861465 -0.0540168019 -0.0912739409
##  [21]  0.0616177365  0.0066884055 -0.0659385449 -0.1590983027 -0.0392326585
##  [26] -0.1468850497 -0.0467161509 -0.1413664326 -0.0240259007 -0.0147998717
##  [31] -0.0280750919 -0.0664797182 -0.1059757587 -0.0372058538 -0.0278251635
##  [36] -0.0341504253 -0.0894378249 -0.1100361615 -0.0628570199 -0.1443427101
##  [41] -0.0902022799 -0.0829628721 -0.0590367723 -0.0515670502 -0.1150610738
##  [46] -0.0965332976 -0.0943285959 -0.0555092219 -0.0732086305 -0.0962907718
##  [51] -0.0725700931 -0.0509452075 -0.0659390811 -0.0008464589 -0.0398504773
##  [56] -0.0606627507 -0.0600523663 -0.0533240823 -0.1321345202 -0.1039725419
##  [61] -0.0244142251 -0.0351216871 -0.0770709466 -0.0510152143 -0.0391089503
##  [66] -0.0941225991  0.0266582335  0.0465720966 -0.0487954721 -0.0654435109
##  [71] -0.1084755013 -0.0592709706 -0.0915116790 -0.0906236988 -0.0932321902
##  [76] -0.0980089287 -0.0984693758 -0.0407218407  0.0104085481 -0.0696932352
##  [81] -0.0910667142 -0.0141206693 -0.0864367954 -0.0662417432 -0.0754077527
##  [86]  0.0350170713 -0.0619622183 -0.0333679471 -0.0746443158 -0.0345850662
##  [91] -0.0499747430 -0.0546158633 -0.0545916682 -0.0679089227 -0.0285587112
##  [96] -0.0451487043  0.0580375476  0.0186410160  0.0049203453  0.0486228461
```

We've got us a numeric vector of as many values as our data had observations--100 in this case. The `pareto_k` values can be used to examine overly-influential cases. See, for example [this discussion on stackoverflow.com](https://stackoverflow.com/questions/39578834/linear-model-diagnostics-for-bayesian-models-using-rstan/39595436) in which several members of the [Stan team](http://mc-stan.org) weighed in. The issue is also discussed in [this paper](https://arxiv.org/abs/1507.04544), in the [loo reference manual](https://cran.r-project.org/web/packages/loo/loo.pdf), and in [this presentation by Vehtari himself](https://www.youtube.com/watch?v=FUROJM3u5HQ&feature=youtu.be&a=). If we explicitly open the [loo package](https://cran.r-project.org/web/packages/loo/index.html), we can use a few convenience functions to leverage `pareto_k` for diagnostic purposes. The `pareto_k_table()` function will categorize the `pareto_k` values and give us a sense of how many values are in problematic ranges.


```r
library(loo)

pareto_k_table(loo_b1)
```

```
## 
## Pareto k diagnostic values:
##                          Count  Pct 
## (-Inf, 0.5]   (good)     98    98.0%
##  (0.5, 0.7]   (ok)        1     1.0%
##    (0.7, 1]   (bad)       1     1.0%
##    (1, Inf)   (very bad)  0     0.0%
```

Happily, most of our cases were in the "good" range. One pesky case [can you guess which one?] was in the "bad" range and another [and can you guess that one, too?] was only "ok." The `pareto_k_ids()` function will tell us which cases we'll want to look at.


```r
pareto_k_ids(loo_b1)
```

```
## [1] 1 2
```

Those numbers correspond to the row numbers in the data, `o`. These are exactly the cases that plagued our second OLS model, `fit1`.

With the simple `plot()` function, we can get a diagnostic plot for the `pareto_k` values.


```r
plot(loo_b1)
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

There they are, cases 1 and 2, lurking in the "bad" and "[just] ok" ranges. We can also make a similar plot with ggplot2. Though it takes a little more work, ggplot2 makes it easy to compare `pareto_k` plots across models with a little faceting.


```r
loo_b0$pareto_k %>%  # The well-behaived data
  as_tibble() %>%
  mutate(i = 1:n()) %>%
  bind_rows(  # The data with two outliers
    loo_b1$pareto_k %>% 
      as_tibble() %>%
      mutate(i = 1:n()) 
  ) %>%
  rename(pareto_k = value) %>%
  mutate(fit = rep(c("fit b0", "fit b1"), each = n()/2)) %>%

  ggplot(aes(x = i, y = pareto_k)) +
  geom_hline(yintercept = c(.5, .7), color = "white") +
  geom_point(alpha = .5) +
  theme(panel.grid = element_blank(),
        axis.title.x = element_text(face = "italic", family = "Times")) +
  facet_wrap(~fit)
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-22-1.png)<!-- -->

So with `b0`--the model based on the well-behaved multivariate normal data, `d`--, all the `pareto_k` values hovered around zero in the "good" range. Things got icky with model `b1`. But we know all that. Let's move forward.

### What do we do with those overly-influential outlying values?

A typical way to handle outlying values is to delete them based on some criterion, such as the Mahalanobis distance, Cook's $D_{i}$, or our new friend, the `pareto_k`. In our next two models, we'll do that. In our `data` arguments, we can use the `slice()` function to omit cases. In model `b1.1`, we simply omit the first and most influential case. In model `b1.2`, we omitted both unduly-influential cases, the values from rows 1 and 2.


```r
b1.1 <- 
  brm(data = o %>% slice(2:100), 
      family = gaussian,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))

b1.2 <- 
  brm(data = o %>% slice(3:100), 
      family = gaussian,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

Here are the summaries for our models based on the `slice[d]` data.


```r
tidy(b1.1) %>% slice(1:3) %>% mutate_if(is.double, round, digits = 2)
```

```
## # A tibble: 3 x 5
##   term        estimate std.error  lower upper
##   <chr>          <dbl>     <dbl>  <dbl> <dbl>
## 1 b_Intercept   0.0600    0.100  -0.100 0.220
## 2 b_x           0.290     0.110   0.100 0.470
## 3 sigma         0.970     0.0700  0.860 1.10
```

```r
tidy(b1.2) %>% slice(1:3) %>% mutate_if(is.double, round, digits = 2)
```

```
## # A tibble: 3 x 5
##   term        estimate std.error  lower upper
##   <chr>          <dbl>     <dbl>  <dbl> <dbl>
## 1 b_Intercept   0.0200    0.0900 -0.130 0.160
## 2 b_x           0.390     0.100   0.230 0.560
## 3 sigma         0.860     0.0600  0.770 0.960
```

They are closer to the true data generating model (i.e., the code we used to make `d`), especially `b1.2`. However, there are other ways to handle the influential cases without dropping them. Finally, we're ready to switch to Student's t!

## Time to leave Gauss for the more general Student's t

Recall that the normal distribution is equivalent to a Student's t with the degrees of freedom parameter, $\nu$, set to infinity. $\nu$ is fixed. Here we'll relax that assumption and estimate $\nu$ from the data. Since $\nu$'s now a parameter, we'll have to give it a prior. For our first Student's t model, we'll estimate $\nu$ with the brms default gamma(2, 0.1) prior.


```r
b2 <- 
  brm(data = o, family = student,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("gamma(2, 0.1)", class = "nu"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

For the next model, we'll switch out that weak gamma(2, 0.1) for a stronger gamma(4, 1). Before fitting the model, it might be useful to take a peek at what that prior looks like. In the plot, below, the orange density in the background is the default gamma(2, 0.1) and the purple density in the foreground is the stronger gamma(4, 1).


```r
ggplot(data = tibble(x = seq(from = 0, to = 60, by = .1)),
       aes(x = x)) +
  geom_ribbon(aes(ymin = 0, 
                  ymax = dgamma(x, 2, 0.1)),
              fill = viridis_pal(direction = 1, option = "C")(5)[4], alpha = 3/4) +
  geom_ribbon(aes(ymin = 0, 
                  ymax = dgamma(x, 4, 1)),
              fill = viridis_pal(direction = 1, option = "C")(5)[2], alpha = 3/4) +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = 0:50) +
  theme(panel.grid = element_blank())
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

So the default prior is centered around values in the 2 to 30 range, but has a long gentle-sloping tail, allowing the model to yield much larger values for $\nu$, as needed. The prior we use below is almost entirely concentrated in the single-digit range. In this case, that will preference Student's t likelihoods with very small $\nu$ parameters and correspondingly thick tails--easily allowing for extreme values.


```r
b3 <- 
  brm(data = o, family = student,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("gamma(4, 1)", class = "nu"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

For our final model, we'll fix the $\nu$ parameter in a `bf()` statement.


```r
b4 <-
  brm(data = o, family = student,
      bf(y ~ 1 + x, nu = 4),
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

Now we've got all those models, we can put all their estimates into one tibble.


```r
# We have to detach MASS and reload tidyverse so we might use tidyverse::select()
detach(package:MASS, unload = T)
library(tidyverse)

b_estimates <-
  tidy(b0) %>%
  bind_rows(tidy(b1)) %>%
  bind_rows(tidy(b1.1)) %>%
  bind_rows(tidy(b1.2)) %>%
  bind_rows(tidy(b2)) %>%
  bind_rows(tidy(b3)) %>%
  bind_rows(tidy(b4)) %>%
  filter(term %in% c("b_Intercept", "b_x")) %>%
  mutate(model = rep(c("b0", "b1", "b1.1", "b1.2", "b2", "b3", "b4"), each = 2)) %>%
  select(model, everything()) %>%
  arrange(term)
```

To get a sense of what we've done, let's take a peek at our models tibble.


```r
b_estimates %>%
  mutate_if(is.double, round, digits = 2)  # This is just to round the numbers
```

```
##    model        term estimate std.error lower upper
## 1     b0 b_Intercept    -0.01      0.09 -0.15  0.13
## 2     b1 b_Intercept     0.12      0.11 -0.05  0.30
## 3   b1.1 b_Intercept     0.06      0.10 -0.10  0.22
## 4   b1.2 b_Intercept     0.02      0.09 -0.13  0.16
## 5     b2 b_Intercept     0.04      0.09 -0.10  0.19
## 6     b3 b_Intercept     0.04      0.09 -0.11  0.19
## 7     b4 b_Intercept     0.04      0.09 -0.11  0.20
## 8     b0         b_x     0.44      0.10  0.28  0.61
## 9     b1         b_x     0.15      0.13 -0.06  0.37
## 10  b1.1         b_x     0.29      0.11  0.10  0.47
## 11  b1.2         b_x     0.39      0.10  0.23  0.56
## 12    b2         b_x     0.35      0.11  0.17  0.53
## 13    b3         b_x     0.36      0.10  0.19  0.53
## 14    b4         b_x     0.36      0.10  0.19  0.53
```

The models differ by their intercepts, slopes, sigmas, and $\nu$s. For the sake of this document, we'll focus on the slopes. Here we compare the different Bayesian models' slopes by their posterior means and 95% intervals in a coefficient plot.


```r
b_estimates %>%
  filter(term == "b_x") %>% # b_Intercept b_x
  
  ggplot(aes(x = model)) +
  geom_pointrange(aes(y = estimate,
                      ymin = lower,
                      ymax = upper),
                  shape = 20) +
  coord_flip(ylim = c(-.2, 1)) +
  labs(title = "The x slope, varying by model",
       subtitle = "The dots are the posterior means and the lines the percentile-based 95% intervals.",
       x = NULL,
       y = NULL) +
  theme(panel.grid = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_text(hjust = 0))
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

You might think of the `b0` slope as the "true" slope. That's the one estimated from the well-behaved multivariate normal data, `d`. That estimate's just where we'd want it to be. The `b1` slope is a disaster--way lower than the others. The slopes for `b1.1` and `b1.2` get better, but at the expense of deleting data. All three of our Student's t models produced slopes that were pretty close to the `b0` slope. They weren't perfect, but, all in all, Students t did pretty okay.

### We need more LOO and more `pareto_k`.

We already have loo objects for our first two models, `b0` and `b1`. Let's get some for models `b2` through `b4`.


```r
loo_b2 <- loo(b2)
loo_b3 <- loo(b3)
loo_b4 <- loo(b4)
```

With a little data wrangling, we can compare our models by how they look in our custom `pareto_k` diagnostic plots. 


```r
loo_b1$pareto_k %>% 
  as_tibble() %>%
  mutate(i = 1:n()) %>%
  bind_rows(
    loo_b2$pareto_k %>% 
      as_tibble() %>%
      mutate(i = 1:n()) 
  ) %>%
  bind_rows(
    loo_b3$pareto_k %>% 
      as_tibble() %>%
      mutate(i = 1:n()) 
  ) %>%
  bind_rows(
    loo_b4$pareto_k %>% 
      as_tibble() %>%
      mutate(i = 1:n()) 
  ) %>%
  rename(pareto_k = value) %>%
  mutate(fit = rep(c("fit b1", "fit b2", "fit b3", "fit b4"), each = n()/4)) %>%

  ggplot(aes(x = i, y = pareto_k)) +
  geom_hline(yintercept = c(.5, .7),
             color = "white") +
  geom_point(alpha = .5) +
  scale_y_continuous(breaks = c(0, .5, .7)) +
  theme(panel.grid = element_blank(),
        axis.title.x = element_text(face = "italic", family = "Times")) +
    facet_wrap(~fit)
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-33-1.png)<!-- -->

Oh man, those Student's t models made our world shiny! In a succession from `b2` through `b4`, each model looked better by `pareto_k`. All were way better than the typical Gaussian model, `b1`. While we're at it, we might compare those for with their LOO values.


```r
compare_ic(loo_b1, loo_b2, loo_b3, loo_b4)
```

```
##          LOOIC    SE
## b1      311.84 31.37
## b2      289.96 23.18
## b3      287.62 20.84
## b4      286.20 20.19
## b1 - b2  21.88 11.85
## b1 - b3  24.22 14.90
## b1 - b4  25.63 15.75
## b2 - b3   2.35  3.18
## b2 - b4   3.76  4.07
## b3 - b4   1.41  0.89
```

In terms of the LOO, `b2` through `b4` were about the same, but all looked better than `b1`. In fairness, though, the standard errors for the difference scores were a bit on the wide side.

If you're new to using information criteria to compare models, you might sit down and soak in [this lecture on the topic](https://www.youtube.com/watch?v=t0pRuy1_190&list=PLDcUM9US4XdM9_N6XUUFrhghGJ4K25bFc&index=8) and [this vignette](https://cran.r-project.org/web/packages/loo/vignettes/loo-example.html) on the LOO in particular. For a more technical introduction, you might check out the references in the loo package's [reference manual](https://cran.r-project.org/web/packages/loo/loo.pdf).

## Let's compare a few Bayesian models

That's enough with coefficients, `pareto_k`, and the LOO. Let's get a sense of the implications of the models by comparing a few in plots.


```r
# These are the values of x we'd like model-implied summaries for
nd <- tibble(x = seq(from = -4, to = 4, length.out = 50))

# We get the model-implied summaries with a big fitted() object
fitted_bs <- 
  # for b0
  fitted(b0, newdata = nd) %>%
  as_tibble() %>%
  # for b1
  bind_rows(
    fitted(b1, newdata = nd) %>%
      as_tibble()
  ) %>% 
  # for b3
  bind_rows(
    fitted(b3, newdata = nd) %>%
      as_tibble()
  ) %>% 
  # we need a model index
  mutate(model = rep(c("b0", "b1", "b3"), each = 50)) %>% 
  # and we'd like to put the x values in there, too
  mutate(x = rep(nd %>% pull(), times = 3))

# The plot
ggplot(data = fitted_bs, 
       aes(x = x)) +
  geom_ribbon(aes(ymin = `2.5%ile`,
                  ymax = `97.5%ile`),
              fill = "grey67") +
  geom_line(aes(y = Estimate),
            color = "grey92") +
  geom_point(data = d %>%
               bind_rows(o) %>%
               bind_rows(o) %>%
               mutate(model = rep(c("b0", "b1", "b3"), each = 100)), 
             aes(x = x, y = y, color = y > 3),
             size = 1, alpha = 3/4) +
  scale_color_manual(values = c("black", viridis_pal(direction = 1, option = "C")(7)[4])) +
  coord_cartesian(xlim = -3:3, 
                  ylim = -3:5) +
  ylab(NULL) +
  theme(panel.grid = element_blank(),
        legend.position = "none") +
  facet_wrap(~model)
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-35-1.png)<!-- -->

For each subplot, the gray band is the 95% interval band and the overlapping light gray line is the posterior mean. Model `b0`, recall, is our baseline comparison model. This is of the well-behaved no-outlier data, `d`, using the good old Gaussian likelihood. Model `b1` is of the outlier data, `o`, but still using the non-robust Gaussian likelihood. Model `b3` uses a robust Student's t likelihood with $\nu$ estimated with the fairly narrow gamma(4, 1) prior. For my money, `b3` did a pretty good job.

## But what if you don't need Student's t?

A while back (before I fell in love with Bayes), I submitted a paper in which we used a robust frequentist estimator (i.e., MLR). One of the reviewers was unfamiliar with robust estimators and asked what would happen if we used a robust estimator when a traditional non-robust estimator (i.e., ML) was good enough. In response, we ran a little simulation study to demonstrate that in such a case, there was little to fear. You’d get the same results within rounding error.

So, what would happen if we used the Student’s t likelihood with our well-behaved `d` data? Will the results look like those from `b0`, the Gaussian model of the `d` data? To answer the question, we’ll first re-fit the Student-t models `b2` through `b4`, this time switching out the nasty outlier `o` data for the nicely-Gaussian `d` data. 


```r
b2_data_d <- 
  brm(data = d, family = student,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("gamma(2, 0.1)", class = "nu"),
                set_prior("cauchy(0, 1)", class = "sigma")))

b3_data_d <- 
  brm(data = d, family = student,
      y ~ 1 + x,
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("gamma(4, 1)", class = "nu"),
                set_prior("cauchy(0, 1)", class = "sigma")))

b4_data_d <-
  brm(data = d, family = student,
      bf(y ~ 1 + x, nu = 4),
      prior = c(set_prior("normal(0, 100)", class = "Intercept"),
                set_prior("normal(0, 10)", class = "b"),
                set_prior("cauchy(0, 1)", class = "sigma")))
```

Here we’ll put the summaries of our three new models and the original Gaussian model, `b0`, into a tibble, wrangle the thing a little, and make a few coefficient plots. 


```r
# The "true" model, b0
tidy(b0) %>% 
  mutate(model = "b0") %>% 
  # Adding the Student's t model with the gamma(2, 0.1) for nu
  bind_rows(
    tidy(b2_data_d) %>% 
      mutate(model = "b2_data_d")
  ) %>% 
  # Adding the Student's t model with the gamma(4, 1) for nu
  bind_rows(
    tidy(b3_data_d) %>% 
      mutate(model = "b3_data_d")
  ) %>% 
  # Adding the Student's t model with fixed `nu = 4`
  bind_rows(
    tidy(b4_data_d) %>% 
      mutate(model = "b4_data_d")
  ) %>% 
  # We'll restrict our plot to the major parameters: the intercept, slope, and sigma
  filter(term != "lp__",
         term != "nu") %>% 
  
  # Finally, plotting!
  ggplot(aes(x = model)) +
  geom_pointrange(aes(y = estimate,
                      ymin = lower,
                      ymax = upper),
                  shape = 20) +
  coord_flip() +
  labs(title = "The d-data intercept, slope and sigma, varying by model",
       subtitle = "The dots are the posterior means and the lines the percentile-based 95% intervals.",
       x = NULL,
       y = NULL) +
  theme(panel.grid = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_text(hjust = 0)) +
  facet_grid(~term, scales = "free_x")
```

![](Robust_linear_regression_with_Student_s_t_distribution_files/figure-html/unnamed-chunk-37-1.png)<!-- -->

Happily, the intercepts and slopes were near equivalent across the four models. If those were your parameters of interest, it seems like switching to a robust Student's t likelihood didn't hurt anything for our well-behaved `d` data. Note, though, that $\sigma$ changed a bit across models. This is because $\sigma$ means something different when you switch from the Gaussian to a lower-$\nu$ Student's t. When you use the Gaussian, $\sigma$ is the standard deviation and $\sigma^2$ is the variance. However, once you use a sub-infinite $\nu$, $\sigma$ is now referred to as the scale parameter and no longer corresponds to the standard deviation. When $\nu$ is quite large, the difference shouldn't be as large. But as your $\nu$s approach 1, the discrepancy between the scale and the standard deviation grows.

Let's compare our models with the LOO.


```r
loo(b0, b2_data_d, b3_data_d, b4_data_d)
```

```
##                        LOOIC    SE
## b0                    256.74 12.94
## b2_data_d             257.98 13.03
## b3_data_d             261.29 13.29
## b4_data_d             264.45 13.41
## b0 - b2_data_d         -1.24  0.57
## b0 - b3_data_d         -4.56  1.63
## b0 - b4_data_d         -7.71  2.30
## b2_data_d - b3_data_d  -3.32  1.07
## b2_data_d - b4_data_d  -6.47  1.75
## b3_data_d - b4_data_d  -3.15  0.69
```

In this case, the “true” model, `b0`, had the lowest LOO value. The values tended to *go up* when we switched to the Student’s t. Among the Student’s t models, the LOO values even seemed to rise a bit the more we constrained $\nu$ to lower values.

In aggregate, switching to a more complex Student’s t model does little harm when you don’t really need it and when your parameters of interest are the intercepts and slopes. If you care about $\sigma$, things get a little messy. If you care about model parsimony, the LOO suggests little is to be gained from adopting the more complicated Student’s t likelihood.

Is there a simple answer? Nope. But now you have choices.

Note. The analyses in this document were done with:

* R         3.4.4
* RStudio   1.1.442
* rmarkdown 1.8
* tidyverse 1.2.1
* viridis   0.4.0
* MASS      7.3-47
* broom     0.4.3
* brms      2.1.9
* rstan     2.17.3
* loo       1.1.0





































