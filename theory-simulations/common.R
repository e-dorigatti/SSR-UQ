library(ggplot2)
library(dplyr)
library(reshape2)
library(tidyr)

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
  df$pow <- (df$lo > 0) | (df$hi < 0)

  cbind(df, as.data.frame(list(...)))
}


solve_glm_breslow <- function(X, y, z, tau, verbose = F, niter = 10) {
  # uses eq. 10 and 11 of Breslow, 1993
  # and fits a Poisson GLM
  
  mu <- y + 1e-2
  eta <- log(mu)
  for(i in 1:niter) {
    yy <- eta + (y - mu) / mu
    if(is.null(tau)) {
      vvi <- diag(1e-9 + mu)
    }
    else {
      vvi <- diag(1e-9 + mu / (1 + mu * tau**2))
    }

    betahat <- solve(t(X) %*% vvi%*% X) %*% t(X) %*% vvi %*% yy
    if(is.null(tau)) {
      bb <- 0 * yy
    }
    else {
      bb <- (diag(1e-9 + tau**2 * mu / (1 + mu * tau**2)) %*% (yy - X %*% betahat))[,1]
    }

    eta <- (X %*% betahat + bb)[,1]
    mu <- exp(eta + z)

    if(verbose) {
      cat(betahat, "\n")
    }
    #plot(log(mu), log(mutrue))
    
  }

  # compute covariance of beta and b
  n <- nrow(X)
  d <- ncol(X)
  if(!is.null(tau)) {
    xx <- cbind(diag(nrow = n), X)
    xwx <- t(xx) %*% diag(1e-9 + mu) %*% xx
    ss <- 0 * xwx
    ss[1:n, 1:n] <- diag(1e-9 + 1 / tau**2)
    xwx <- xwx + ss
    xwxi <- solve(xwx)

    covariance <- xwxi[(n+1):(n+d), (n+1):(n+d)]
    bhat_variance <- diag(xwxi[1:n, 1:n])
    b_beta_covariance <- xwxi[1:n, (n+1):(n+d)]
  }
  else {
    covariance <- solve(t(X) %*% vvi %*% X)
    bhat_variance <- rep(0, n)
    b_beta_covariance <- matrix(rep(0, n * d), nrow = n)
  }

  list(
    # return initial data for convenience
    X = X,
    y = y,
    z = z,
    tau = tau,

    # predictions
    etahat = (X %*% betahat + bb + z)[,1],
    yhat = mu,

    # inference on beta
    betahat = betahat[,1],
    covariance = covariance,

    # inference on b
    bhat = bb,
    bhat_variance = bhat_variance,
    
    # both
    b_beta_covariance = b_beta_covariance
  )
}


get_structured <- function(df) {
  cols <- colnames(df)
  as.matrix(df[, cols[grep("str_.*", cols)]])
}


solve_glm_breslow_for_ensemble <- function(df, z_type, tau_type) {
  stopifnot(!("ensemble_id" %in% colnames(df)) | length(unique(df$ensemble_id)) > 1)
  stopifnot(!("subset" %in% colnames(df)) | length(unique(df$subset)) == 1)

  dnn <- df %>% group_by(sample) %>% summarise(
    y = mean(y),
    z = mean(pred_uns),
    ff = mean(f),
    taupred = sd(pred_uns),
    tautrue = sqrt(mean((pred_uns - f)**2)),
    eta = mean(eta),
    dnn_pred = mean(preds),
    dnn_pred_var = var(preds),
  )

  df1 <- df %>% filter(ensemble_id == 1)
  x_str <- get_structured(df1)

  if(tau_type == "ens notau") {
    tau <- NULL
  }
  else if(tau_type == "ens taupred") {
    tau <- dnn$taupred
  }
  else if(tau_type == "ens tautrue") {
    tau <- dnn$tautrue
  }
  else {
    stop()
  }

  if(z_type == "ens") {
    z <- dnn$z
  }
  else if(z_type == "data") {
    z <- dnn$ff
  }
  else {
    stop()
  }

  res <- solve_glm_breslow(x_str, dnn$y, z, tau)
  res$f <- dnn$ff
  res$eta_true <- dnn$eta
  res$dnn_pred <- dnn$dnn_pred
  res$dnn_pred_var <- dnn$dnn_pred_var
  res$beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])
  res$tau_type = tau_type
  res$z_type = z_type
  res
}


solve_glm_breslow_for_network <- function(df, network_num = NULL) {
  stopifnot(!("subset" %in% colnames(df)) | length(unique(df$subset)) == 1)
  if("ensemble_id" %in% colnames(df)) {
    if(is.null(network_num)) {
      stopifnot(length(unique(df$ensemble_id)) == 1)
    }
    else {
      df <- df %>% filter(ensemble_id == network_num)
      stopifnot(length(df) > 0)
    }
  }

  cols <- colnames(df)
  x_str <- as.matrix(df %>% dplyr::select(cols[grep("str_.*", cols)]))
  res <- solve_glm_breslow(x_str, df$y, df$pred_uns, NULL)
  res$f <- df$f
  res$eta_true <- df$eta
  res$dnn_pred <- df$preds
  res$dnn_pred_var <- 0
  res$beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])
  res
}


binomial_ci <- function(samples, alpha) {
  # computes the confidence interval bound corresponding to the given alpha for
  # a binomial random variable with given samples using Jeffrey's formula
  # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval

  x <- sum(samples)
  n <- length(samples)
  qbeta(alpha, x + 0.5, n - x + 0.5)
}

continuous_ci <- function(x, alpha, prediction_interval = FALSE) {
  m <- mean(x)
  if(prediction_interval) {
    s <- sd(x)
  }
  else {
    s <- sd(x) / sqrt(length(x))
  }
  m + qt(alpha, length(x)) * s
}
