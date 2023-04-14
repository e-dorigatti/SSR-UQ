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


make_cov_df <- function(n, beta_true, beta_hat, beta_var, level = 0.85, ...) {
  cr <- qt(1-(1-level)/2, n)
  df <- data.frame(
    beta = beta_true,
    est = beta_hat,
    err = sqrt(beta_var),
    lo = beta_hat - cr * sqrt(beta_var),
    hi = beta_hat + cr * sqrt(beta_var)
  )
  df$cov <- (df$lo <= df$beta) & (df$beta <= df$hi)
  df$pow <- (df$lo > 0) | (df$hi < 0)

  cbind(df, as.data.frame(list(...)))
}


### simulation #####


run_sim <- function(n, d, sigma, tau_sd, nboot = 25, ...) {
  cat("runinng sim for", n, d, sigma, tau_sd, nboot, "\n")
  stopifnot(n > d)
  stopifnot(sigma > 0)
  stopifnot(tau_sd > 0)
  stopifnot(nboot > 0)

  res_params <- c()
  res_preds <- c()

  for(j in 1:as.integer(sqrt(nboot))) {
    beta <- rnorm(d, sd = sqrt(d))
    f <- rnorm(n)

    for(i in 1:as.integer(sqrt(nboot))) {
      # generate data with linearly dependent features
      X <- matrix(rnorm(n * d), nrow = n) # %*% matrix(rnorm(d * d), nrow = d)
      eps <- rnorm(n, sd = sigma)
      t <- X %*% beta + f
      y <- t + eps

      # auxiliary matrices
      xp <- solve(t(X) %*% X)  # covariance
      xi <- xp %*% t(X)        # pseudo-inverse
      xh <- X %*% xi           # hat
      In <- diag(rep(1, n))    # n x n identity

      #tau <- exp(rnorm(  # log-normal with mean 1 and sd tau_sd
      #  n, mean = -log(tau_sd**2 + 1), sd = sqrt(log(tau_sd**2 + 1))
      #))
      #z <- rnorm(n, mean = f, sd = tau)
      # tau <- diag(tau)

      # simulate DNN, inducing dependence between features and uncertainty
      A <- matrix(rnorm(n * n, mean = 0.75), ncol = n)
      Gc <- t(A) %*% A
      Gc <- diag(sqrt(diag(1/Gc))) %*% Gc %*% diag(sqrt(diag(1/Gc)))
      tau <- tau_sd * Gc
      z <- MASS::mvrnorm(mu = f, Sigma = tau)

      beta_hat <- xi %*% (y - z)

      # estimate predictions
      yhat <- X %*% beta_hat + z

      # estimate params
      resid <- y - yhat
      sigma_tau <- sum(resid**2) / (n - d)

      # compare methods for prediction variance
      res_preds <- rbind(
        res_preds,
        make_cov_df(
          # exact formula from theory
          n, t, yhat,
          diag(sigma**2 * xh + (xh - In) %*% tau %*% (xh - In)),
          method = "exact", rep = i, sample = 1:n
        ),
        make_cov_df(
          # asymptotic normality result
          n, t, yhat,
          diag(sigma**2 * xh + xh %*% tau %*% xh + diag(tau**2)),
          method = "alternative", rep = i, sample = 1:n
        ),
        make_cov_df(
          # approximating tau as constant
          n, t, yhat,
          diag(sigma**2 * xh + (xh - In) %*% (mean(diag(tau)) * In) %*% (xh - In)),
          method = "const_dnn", rep = i, sample = 1:n
        ),
        make_cov_df(
          # this is what you'd do in a normal LM
          n, t, yhat,
          diag(sum(resid**2) / (n - d) * xh),
          method = "lm", rep = i, sample = 1:n
        ),
        make_cov_df(
          # this is as above but using the residuals instead of the true variance
          n, t, yhat,
          diag((xh - In) %*% diag(resid[,1]**2) %*% (xh - In)),
          method = "lm_2_r", rep = i, sample = 1:n
        )
      )

      # compare methods for parameter variance
      res_params <- rbind(
        res_params,
        make_cov_df(
          n, beta, beta_hat,
          diag(xi %*% (sigma**2 * In + tau) %*% t(xi)),
          method = "true_all", rep = i, coef = 1:d
        ),
        make_cov_df(
          n, beta, beta_hat, diag(sigma**2 * xp),
          method = "true_no_tau", rep = i, coef = 1:d
        ),
        make_cov_df(
          n, beta, beta_hat,
          diag((sigma**2 + mean(diag(tau))) * xp),
          method = "true_tau_mean", rep = i, coef = 1:d
        ),
        make_cov_df(
          n, beta, beta_hat,
          diag(sigma_tau * xp),
          method = "lm_sigma_est", rep = i, coef = 1:d
        )
      )
    }
  }
  list(
    res_params = cbind(res_params, as.data.frame(list(...))),
    res_preds = cbind(res_preds, as.data.frame(list(...)))
  )
}


run_grid <- function() {
  # build grid
  grd <- c()
  d <- 4
  for(n in c(5, 10, 50, 100)) { #, 250, 500)) {
    for(sigma in c(0.25, 1, 4)) {
      for(tau in c(0.25, 1, 4)) {
        grd <- rbind(grd, list(n, sigma, tau))
      }
    }
  }

  # compute in parallel
  library(parallel)
  rr <- mclapply(1:nrow(grd), function(i) {
    run_sim(grd[[i,1]], d, grd[[i,2]], grd[[i,3]], tau, nboot = 25,  # 100
            noise_sd = grd[[i,2]], dnn_uq = grd[[i,3]], n_samples = grd[[i,1]])
  }, mc.cores = 6)

  # aggregate results
  res_params <- c()
  res_preds <- c()
  for(i in 1:nrow(grd)) {
    res_params <- rbind(res_params, rr[[i]]$res_params)
    res_preds <- rbind(res_preds, rr[[i]]$res_preds)
  }

  list(res_params = res_params, res_preds = res_preds)
}

res <- run_grid()
saveRDS(res, "simulation-ols-fullcov.RDS")

res <- readRDS("simulation-ols-fullcov.RDS")

# *** paper plot ***

res$res_params %>%
  group_by(n_samples, noise_sd, dnn_uq, method) %>%
  summarise(
    rbind(
      data.frame(
        variable = "coverage", value = mean(cov), ymin = binomial_ci(cov, 0.05), ymax = binomial_ci(cov, 0.95)
      ),
      data.frame(
        variable = "power", value = mean(pow), ymin = binomial_ci(pow, 0.05), ymax = binomial_ci(pow, 0.95)
      )
    )
  ) %>%
  filter(
    method %in% c("true_all", "true_no_tau", "lm_sigma_est") &
      variable == "coverage" &
      n_samples < 110 #& dnn_uq != 1 & noise_sd != 1
  ) %>%
  mutate(
    Method = factor(
      method,
      levels = c("true_all", "true_no_tau", "lm_sigma_est"),
      labels = c("True param. (both)", "True param. (no gamma)", "Residuals only")
    ),
    dnn_uq = factor(
      dnn_uq, levels=c(0.25, 1, 4), labels=paste("gamma:", c(0.5, 1, 2))
    ),
    noise_sd = factor(
      noise_sd, levels=c(0.25, 1, 4), labels=paste("sigma:", c(0.5, 1, 2))
      )
  ) %>%
  ggplot(aes(
    x = as.factor(n_samples), y = value,
    group = Method, color = Method, shape = Method,
    ymin = ymin, ymax = ymax)
  ) +
  geom_point(position = position_dodge(0.8)) +
  geom_errorbar(position = position_dodge(0.8)) +
  geom_hline(yintercept = 0.85, linetype = "dashed") +
  facet_grid(dnn_uq ~ noise_sd) +
  theme_bw() +
  labs(x = "Train. Size", y = "Coverage of beta")
ggsave("/tmp/cor1.pdf", width = 9, height = 4)


