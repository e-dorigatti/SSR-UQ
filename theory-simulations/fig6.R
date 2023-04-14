source("commons.R")

rows <- readRDS("poisson-dnn.RDS")


# inference of y before and after -----------------------------------------
level <- 0.85

df <- rows %>% filter(subset == "test" & dset == 1)

rr <- solve_glm_breslow_for_ensemble(df, "ens", "ens tautrue")
rr <- solve_glm_breslow_for_network(df, 1)

norm_lik <- function(pred, mu, var) {
  # sample likelihood of a multivariate normal with diagonal covariance
  log(2 * pi) + (pred - mu)**2 / (2 * var / 2) + log(var) / 2
}

comp_cov <- function(pred, true, var) {
  cr <- qt(1-(1-level)/2, length(pred))
  (pred - cr * sqrt(var) <= true) & (true <= pred + cr * sqrt(var))
}

summarise_glmm <- function(rr) {
  if(class(rr) == "try-error") {
    return(data.frame())
  }

  if(is.null(rr$tau)) {
    rr$tau <- 0
  }

  f_hat <- rr$z + rr$bhat
  f_hat_var <- rr$bhat_variance

  eta_hat_var <- (
    #diag(rr$X %*% rr$covariance %*% t(rr$X))
    diag(rr$X %*% solve(t(rr$X) %*% diag(rr$yhat) %*% rr$X) %*% t(rr$X))
    + rr$bhat_variance
    + 2 * diag(rr$b_beta_covariance %*% t(rr$X))
  )
  #if(!is.null(rr$tau_type) && rr$tau_type == "ens tautrue") {
  #eta_hat_var <- eta_hat_var - diag(rr$X %*% rr$beta_true %*% t(rr$beta_true) %*% t(rr$X))
  #}

  data.frame(
    f_sqerr = (rr$f - f_hat)**2,
    eta_sqerr = (rr$etahat - rr$eta_true)**2,

    f_likel = norm_lik(f_hat, rr$f, f_hat_var),
    eta_likel = norm_lik(rr$etahat, rr$eta_true, eta_hat_var),

    f_cov = comp_cov(f_hat, rr$f, f_hat_var),
    eta_cov = comp_cov(rr$etahat, rr$eta_true, eta_hat_var)
  )
}

summarise_ens <- function(df) {
  dnn <- df %>% group_by(sample) %>% summarise(
    f_true = mean(f),
    f_pred = mean(pred_uns),
    f_pred_var = var(pred_uns),

    eta_true = mean(eta),
    eta_pred = mean(log(preds)),
    eta_pred_var = var(log(preds))
  )

  data.frame(
    f_sqerr = (dnn$f_true - dnn$f_pred)**2,
    eta_sqerr = (dnn$eta_true - dnn$eta_pred)**2,

    f_likel = norm_lik(dnn$f_pred, dnn$f_true, dnn$f_pred_var),
    eta_likel = norm_lik(dnn$eta_pred, dnn$eta_true, dnn$eta_pred_var),

    f_cov = comp_cov(dnn$f_pred, dnn$f_true, dnn$f_pred_var),
    eta_cov = comp_cov(dnn$eta_pred, dnn$eta_true, dnn$eta_pred_var)
  )
}


sqerrs <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    df <- cur_data()

    rbind(
      summarise_glmm(try({solve_glm_breslow_for_ensemble(df, "ens", "ens tautrue")})) %>% mutate(method = "Exact"),
      summarise_glmm(try({solve_glm_breslow_for_ensemble(df, "ens", "ens taupred")})) %>% mutate(method = "GLMM (Ours)"),
      summarise_ens(df) %>% mutate(method = "Empirical"),
      summarise_glmm(try({solve_glm_breslow_for_network(df, 1)})) %>% mutate(method = "None")
    )
  })


plot_data <- sqerrs %>%
  melt(id.vars = c("gamma", "n_samples", "dset", "method")) %>%
  separate(
    variable, c("varb", "meas"), "_"
  )


plot_data %>%
  filter(value < 50) %>%
  #filter(meas == "cov") %>%
  filter(method != "None") %>%
  group_by(gamma, n_samples, method, varb, meas) %>%
  summarise(
    mc = mean(value), cl = continuous_ci(value, 0.05), ch = continuous_ci(value, 0.95)
  ) %>%
  ggplot(aes(x = varb, y = mc, color = method)) +
  geom_point(position = position_dodge(0.25)) +
  geom_errorbar(aes(ymin = cl, ymax = ch), position = position_dodge(0.25)) +
  geom_hline(yintercept = level) +
  #scale_y_log10() +
  facet_grid(meas ~ gamma + n_samples, scales = "free")


# paper plot

dset_avg <- plot_data %>%
  filter(value < 50) %>%
  filter(meas == "sqerr") %>%
  group_by(gamma, n_samples, method, varb, meas, dset) %>%
  summarise(avgg = sqrt(mean(value)))

exact_avg <- dset_avg %>% filter(method == "GLMM (Ours)")

rel_avg <- dset_avg %>%
  merge(exact_avg, by = c("gamma", "n_samples", "varb", "meas", "dset")) %>%
  mutate(
    rel_avg = 100 * (avgg.x / avgg.y - 1),
    method = method.x
  )

increa <- rel_avg %>%
  group_by(gamma, n_samples, method, varb, meas) %>%
  summarise(
    mc = mean(rel_avg),
    #cl = sqrt(quantile(avg, 0.05)[[1]]), #sqrt(continuous_ci(value, 0.05)),
    #ch = sqrt(quantile(avg, 0.95)[[1]]), #sqrt(continuous_ci(value, 0.95))
    cl = continuous_ci(rel_avg, 0.05),
    ch = continuous_ci(rel_avg, 0.95)
  )


ggplot(increa, aes(x = method, y = mc, color = varb, shape = varb)) +
  geom_point(position = position_dodge(0.25)) +
  geom_errorbar(aes(ymin = cl, ymax = ch), position = position_dodge(0.25), width = 0) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  #scale_y_log10() +
  labs(x = "", y = "Relative RMSE difference %") +
  scale_x_discrete(limits=c("None", "Empirical", "GLMM (Ours)", "Exact")) +
  facet_grid(gamma ~ n_samples, scales = "free", labeller = as_labeller(c(
    `75` = "Train. size: 75", `500` = "Train. size: 500", `2000` = "Train. size: 2000",
    `0` = "tau: 0", `0.5` = "tau: 1/2", `2` = "tau: 2"
  ))) +
  theme_bw() +
  scale_color_hue("Variable", labels = c(expression(eta), "f")) +
  scale_shape_discrete("Variable", labels = c(expression(eta), "f")) +
  theme(axis.text.x=element_text(angle=-45))
ggsave("figures/pred-rmse.pdf", width = 6, height = 4)



