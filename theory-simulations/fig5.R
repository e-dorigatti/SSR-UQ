source("common.R")

rows <- readRDS("poisson-dnn.RDS")


# compare coverage of beta with coverage of f -----------------------------
level <- 0.85

#df <- rows %>% filter(dset == 1 & subset == "test")
#tau_type <- "ens taupred"

compute_ensemble_cov <- function(df, tau_type) {
  res <- try({
    solve_glm_breslow_for_ensemble(df, "ens", tau_type)
  })
  if(class(res) == "try-error") {
    data.frame()
  }
  else {
    cr <- qt(1-(1-level)/2, max(df$sample))

    # coverage of beta
    beta_sd <- sqrt(diag(res$covariance))
    beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])
    cov <- (res$betahat - cr * beta_sd <= beta_true) & (beta_true <= res$betahat + cr * beta_sd)
    pow <- (res$betahat - cr * beta_sd > 0) | (res$betahat + cr * beta_sd < 0)

    # coverage of f
    #f_hat <- res$z + res$bhat
    #f_hat_sd <- sqrt(res$tau**2 + res$bhat_variance) # + mean((res$z + res$bhat - res$f)**2))
    #f_cov <- (f_hat - cr * f_hat_sd <= res$f) & (res$f <= f_hat + cr * f_hat_sd)

    data.frame(
      coef = 1:length(cov),
      beta_covered = cov,
      power = pow
      #f_coverage = mean(f_cov)
    )
  }
}


# coverage by our GLMM
bc_glmm <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    compute_ensemble_cov(cur_data(), "ens taupred")
  }) %>%
  filter(!is.na(beta_covered))


# coverage with exact tau
bc_exact <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    compute_ensemble_cov(cur_data(), "ens tautrue")
  }) %>%
  filter(!is.na(beta_covered))


# coverage without UQ
bc_net <- rows %>%
  filter(subset == "test") %>%
  filter(ensemble_id == 1) %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    df <- cur_data()

    res <- try({
      solve_glm_breslow_for_network(df)
    })
    if(class(res) == "try-error") {
      data.frame()
    }
    else {
      cr <- qt(1-(1-level)/2, max(df$sample))
      beta_sd <- sqrt(diag(res$covariance))
      beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])

      cov <- (res$betahat - cr * beta_sd <= beta_true) & (beta_true <= res$betahat + cr * beta_sd)
      pow <- (res$betahat - cr * beta_sd > 0) | (res$betahat + cr * beta_sd < 0)

      data.frame(
        coef = 1:length(cov),
        beta_covered = cov,
        power = pow
      )
    }
  }) %>%
  filter(!is.na(beta_covered))


# coverage empirical
bc_ens <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    dd <- cur_data() %>% filter(sample == 1)

    xx <- lapply(1:4, function(i) {
      make_cov_df(
        n = nrow(dd),
        beta_true = mean(dd[[paste0("true_beta_", i)]]),
        beta_hat = mean(dd[[paste0("fitted_beta_", i)]]),
        beta_var = var(dd[[paste0("fitted_beta_", i)]]),
        level = level,
        coef = i
      )
    })

    do.call(rbind, xx)
  }) %>%
  filter(!is.na(cov))


# compute empirical coverage of f by the ensemble
fc <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    df <- cur_data()

    ps <- df %>% group_by(sample) %>% summarise(
      z = mean(pred_uns),
      t = sd(pred_uns),
      f = mean(f)
    )

    make_cov_df(
      length(unique(df$ensemble_id)),
      ps$f, ps$z, ps$t**2,
      level = 0.85,
      aa = 2
    ) %>% summarise(
      f_coverage = mean(cov),
      #fcl = binomial_ci(cov, 0.05),
      #fch = binomial_ci(cov, 0.95)
    )
  })


lm_data <- rbind(
    bc_glmm %>% mutate(method = "GLMM (Ours)"),
    bc_exact %>% mutate(method = "Exact"),
    bc_net %>% mutate(method = "None"),
    bc_ens %>% rename(beta_covered = cov, power = pow) %>%
      select(-c(beta, est, err, lo, hi)) %>%
      mutate(method = "Ensemble")) %>%
  merge(fc)

#lm_data$f_cov_bin <- cut(lm_data$f_coverage, seq(0, 1, 0.1))

lm_data$f_cov_bin <- cut(lm_data$f_coverage, c(0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), include.lowest = T)

lm_data$method <- factor(lm_data$method, levels = c("None", "Ensemble", "GLMM (Ours)", "Exact"))
cmod <- glm(beta_covered ~ as.factor(gamma) + as.factor(n_samples) + as.factor(coef) + f_coverage + method,
            family = "binomial", data = lm_data)
summary(cmod)


plot_data <- lm_data %>%
  melt(measure.vars = c("beta_covered", "power")) %>%
  group_by(gamma, n_samples, f_cov_bin, variable, method) %>%
  summarise(
    avg = mean(value),
    cl = binomial_ci(value, 0.05),
    ch = binomial_ci(value, 0.95)
  )
plot_data[plot_data$variable == "beta_covered", "target"] <- level


# plot all data
ggplot(plot_data, aes(x = f_cov_bin, y = avg, color = method)) +
  geom_point(position = position_dodge(0.5), shape = "plus") +
  geom_errorbar(aes(ymin = cl, ymax = ch), width = 0.15, position = position_dodge(0.5)) +
  geom_hline(aes(yintercept = target), linetype = "dashed") +
  facet_grid(variable + gamma ~ n_samples)

# paper plot
plot_data %>%
  filter(variable == "beta_covered") %>%
  ggplot(aes(x = f_cov_bin, y = avg, color = method, shape = method)) +
  geom_point(position = position_dodge(0.25)) +
  geom_errorbar(aes(ymin = cl, ymax = ch), width = 0, position = position_dodge(0.25)) +
  geom_hline(aes(yintercept = target), linetype = "dashed") +
  facet_grid(gamma ~ n_samples, labeller = as_labeller(c(
     `75` = "Train. size: 75", `500` = "Train. size: 500", `2000` = "Train. size: 2000",
     `0` = "tau: 0", `0.5` = "tau: 1/2", `2` = "tau: 2"
   ))) +
  labs(x = "Empirical overage of f by the ensemble", y = "Coverage of beta",
       color = "Uncertainty", shape = "Uncertainty") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=-45))
ggsave("figures/beta-cov-by-f.pdf", width = 6, height = 4)
