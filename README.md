The purpose of this document is to demonstrate the advantages of the Student's t distribution for regression with outliers, particularly within a [Bayesian framework](https://www.youtube.com/channel/UCNJK6_DZvcMqNSzQdEkzvzA/playlists).

In this document, I’m presuming you are familiar with linear regression, familiar with the basic differences between frequentist and Bayesian approaches to fitting regression models, and have a sense that the issue of outlier values is a pickle worth contending with. All code in is [R](https://www.r-bloggers.com/why-use-r-five-reasons/), with a heavy use of the [tidyverse](http://style.tidyverse.org) and the [brms](https://cran.r-project.org/web/packages/brms/index.html) package.

**Here's the deal**: The Gaussian likelihood, which is typical for regression models, is sensitive to outliers. The normal distribution is a special case of Student's t distribution with *nu* (the degree of freedom parameter) set to infinity. However, when *nu* is small, Student's t distribution is more robust to multivariate outliers. For more on the topic, see [Gelman & Hill (2007, chapter 6)](http://www.stat.columbia.edu/~gelman/arm/) or [Kruschke (2014, chapter 16)](https://sites.google.com/site/doingbayesiandataanalysis/).

In this document, we demonstrate how vunlerable the Gaussian likelihood is to outliers and then compare it too different ways of using Student's t likelihood for the same data.
