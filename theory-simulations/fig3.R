library(ggplot2)
library(dplyr)
library(reshape2)


binomial_ci <- function(samples, alpha) {
  # computes the confidence interval bound corresponding to the given alpha for
  # a binomial random variable with given samples using Jeffrey's formula
  # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval

  x <- sum(samples)
  n <- length(samples)
  qbeta(alpha, x + 0.5, n - x + 0.5)
}


continuous_ci <- function(x, alpha) {
  m <- mean(x)
  s <- sd(x) / sqrt(length(x))
  m + qt(alpha, length(x)) * s
}


make_cov_df <- function(n, beta_true, beta_hat, beta_var, level = 0.95, ...) {
  cr <- qt(1-(1-level)/2, n)
  df <- data.frame(
    beta = beta_true,
    est = beta_hat,
    err = sqrt(beta_var),
    lo = beta_hat - cr * sqrt(beta_var),
    hi = beta_hat + cr * sqrt(beta_var)
  )
  df$cov <- (df$lo <= df$beta) & (df$beta <= df$hi)

  cbind(df, as.data.frame(list(...)))
}


fit_poisson_glm <- function(X, y, offset, ...) {
  fit_glm(
    X, y, offset,
    family_log_density = function(yhat, y) { dpois(y, yhat, log = TRUE) },
    link = log,
    link_inverse = exp,
    link_derivative = function(x) { 1 / (x + 1e-8) },
    V = function(mu) { mu },
    ...
  )
}

fit_glm <- function(X, y, offset, family_log_density, link, link_inverse, link_derivative, V,
                    max_iter = 1000, stop_epsilon = 1e-4, verbose = FALSE) {
  mu <- y + 0.5
  eta <- link(mu) - offset
  b <- rnorm(ncol(X), sd = 1 / sqrt(ncol(X)))
  prev_nll <- NULL
  nll_delta <- NA

  for(i in 1:max_iter) {
    # here we use "Fisher scoring", using expected instead of empirical Hessian
    z <- link_derivative(mu) * (y - mu) + eta
    w <- diag(1 / (link_derivative(mu)**2 * V(mu)))

    b <- (solve(t(X) %*% w %*% X + diag(1e-12, ncol(X))) %*% t(X) %*% w %*% z)[,1]
    eta <- (X %*% b)[,1]
    mu <- link_inverse(eta + offset)

    nll <- sum(family_log_density(mu, y))
    if(i > 1) {
      nll_delta <- nll - prev_nll
    }
    prev_nll <- nll

    if(verbose) {
      cat("i:", i, "NLL:", nll, "NLL delta:", nll_delta, "\n", sep = "\t")
    }

    if(i > 1 && nll_delta < stop_epsilon) {
      break
    }
  }

  list(
    estimate = b,
    variance = solve(t(X) %*% w %*% X + diag(1e-12, ncol(X))),
    nll = nll
  )
}

solve_glm_breslow <- function(X, y, z, tau) {
  # uses eq. 10 and 11 of Breslow, 1993 (Z=identity, D=tau)
  # tau here is the full covariance matrix

  mu <- y + 1e-2
  eta <- log(mu)
  for(i in 1:10) {
    yy <- eta + (y - mu) / mu
    vvi <- solve(1e-9 + diag(1/mu) + tau)

    betahat <- solve(t(X) %*% vvi%*% X) %*% t(X) %*% vvi %*% yy
    bb <- (tau %*% vvi) %*% (yy - X %*% betahat)

    eta <- (X %*% betahat + bb)[,1]
    mu <- exp(eta + z)

    #plot(log(mu), log(mutrue))
    #cat(betahat, "\n")
  }

  list(
    # return initial data for convenience
    X = X,
    y = y,
    z = z,
    tau = tau,

    # predictions
    eta = (X %*% betahat + bb + z)[,1],
    yhat = mu,

    # inference on beta
    betahat = betahat[,1],
    covariance = solve(t(X) %*% vvi %*% X),

    # inference on b
    bhat = bb[,1],
    bhat_variance = tau**4 * diag(  # FIXME
      vvi %*% X %*% solve(t(X) %*% vvi %*% X) %*% t(X) %*% t(vvi)
    )
  )
}


n <- 100
d <- 4
rows_beta <- c()
rows_f <- c()
for(tgf in c(0.01, 0.1, 0.25, 0.5, 1, 2, 4, 8)) {
  cat(tgf, "\n")
  for(i in 1:250) {
    X <- matrix(rnorm(n * d), nrow = n) #%*% matrix(rnorm(d * d), nrow = d)

    gamma <- 1
    kappa <- 1
    beta <- rnorm(d, sd = kappa)
    f <- rnorm(n, sd = gamma)

    # sample z with random full-rank covariance (Gc are pearson correlations)
    A <- matrix(rnorm(n * n, mean = 0.75), ncol = n)
    Gc <- t(A) %*% A
    Gc <- diag(sqrt(diag(1/Gc))) %*% Gc %*% diag(sqrt(diag(1/Gc)))
    tau <- tgf * Gc
    z <- MASS::mvrnorm(mu = f, Sigma = tau)

    mutrue <- exp(X %*% beta + f)
    y <- rpois(n, mutrue)

    try({
      level <- 0.85
      cr <- qt(1-(1-level)/2, n)

      res <- solve_glm_breslow(X, y, z, tau)
      mu <- exp(X %*% res$betahat + z)
      #plot(log(mu), log(mutrue))
      sd_beta <- sqrt(diag(res$covariance))
      cov_br <- (res$betahat - cr * sd_beta <= beta) & (beta <= res$betahat + cr * sd_beta)
      br_zero <- (res$betahat - cr * sd_beta <= 0) & (0 <= res$betahat + cr * sd_beta)
      sd_f <- sqrt(tau**2 + res$bhat_variance)
      fcov_br <- (res$bhat + res$z - cr * sd_f <= f) & (f <= res$bhat + res$z + cr * sd_f)

      resg <- fit_poisson_glm(X, y, z)
      mu_glm <- exp(X %*% resg$estimate + z)
      #plot(log(mu_glm), log(mutrue))
      sd_beta <- sqrt(diag(resg$variance))
      cov_glm <- (resg$estimate - cr * sd_beta <= beta) & (beta <= resg$estimate + cr * sd_beta)
      glm_zero <- (resg$estimate - cr * sd_beta <= 0) & (0 <= resg$estimate + cr * sd_beta)
      fcov_glm <- (z - cr * tau <= f) & (f <= z + cr * tau)

      rows_beta <- rbind(
        rows_beta,
        data.frame(coef = 1:d, tgf = tgf, type = "breslow", meas = "coverage", value = cov_br),
        data.frame(coef = 1:d, tgf = tgf, type = "breslow", meas = "power", value = !br_zero),
        data.frame(coef = 1:d, tgf = tgf, type = "breslow", meas = "rmse", value = sqrt(mean((log(mu) - log(mutrue))**2))),
        data.frame(coef = 1:d, tgf = tgf, type = "glm", meas = "coverage", value = cov_glm),
        data.frame(coef = 1:d, tgf = tgf, type = "glm", meas = "power", value = !glm_zero),
        data.frame(coef = 1:d, tgf = tgf, type = "glm", meas = "rmse", value = sqrt(mean((log(mu_glm) - log(mutrue))**2))),
        data.frame(coef = 1:d, tgf = tgf, type = "glm", meas = "cor(z,f)", value = cor(z, f))
      )

      rows_f <- rbind(
        rows_f,
        data.frame(sample = 1:n, tgf = tgf, type = "breslow", meas = "coverage", value = fcov_br),
        data.frame(sample = 1:n, tgf = tgf, type = "breslow", meas = "error", value = abs(f - res$bhat - res$z)),
        data.frame(sample = 1:n, tgf = tgf, type = "glm", meas = "coverage", value = fcov_glm),
        data.frame(sample = 1:n, tgf = tgf, type = "glm", meas = "error", value = abs(f - z))
      )
    })
  }
}

# coverage of beta
rows_beta %>%
  group_by(tgf, type, meas, coef) %>%
  summarise(avg = mean(value), se = sd(value) / sqrt(n())) %>%
  ggplot(aes(x = as.factor(tgf), y = avg, color = type, group = coef)) +
  geom_point(position = position_dodge(0.25)) +
  geom_errorbar(aes(ymin = avg - se, ymax = avg + se), position = position_dodge(0.25)) +
  facet_grid(meas ~ ., scales = "free_y")


# coverage of f
rows_f %>%
  group_by(tgf, type, meas) %>%
  summarise(avg = mean(value), se = sd(value) / sqrt(n())) %>%
  ggplot(aes(x = as.factor(tgf), y = avg, color = type)) +
  geom_point() +
  geom_errorbar(aes(ymin = avg - se, ymax = avg + se)) +
  facet_grid(meas ~ ., scales = "free_y")


# *** paper plot ***

rows_beta %>%
  filter(tgf %in% c(0.01, 0.1, 1, 2) & meas %in% c("coverage", "power")) %>%
  group_by(tgf, type, meas) %>%
  summarise(avg = mean(value), cl = binomial_ci(value, 0.05), ch = binomial_ci(value, 0.95)) %>%
  mutate(target_level = ifelse(meas == "coverage", level, NA)) %>%
  ggplot(aes(x = as.factor(tgf), y = avg, color = type)) +
  geom_point(position = position_dodge(0.25)) +
  geom_errorbar(aes(ymin = cl, ymax = ch), position = position_dodge(0.25), width = 0.1) +
  geom_hline(aes(yintercept = target_level, color = "breslow")) +
  facet_grid(meas ~ ., scales = "free", labeller = as_labeller(c(coverage = "Coverage", power = "Power"))) +
  scale_color_manual(name = "Accounting for\nDNN uncertainty",  values = c("breslow" = "#E69F00", "glm" = "#56B4E9"),
                     labels = c("Yes", "No")) +
  labs(x = "DNN uncertainty", y = "Value") +
  theme_bw()
ggsave("figures/glm.pdf", width = 5, height = 3.5)
