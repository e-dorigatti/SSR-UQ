source("common.R")

rows <- readRDS("poisson-dnn.RDS")


# plot coverage and power -------------------------------------------------

level <- 0.85

# of individial networks
cov_dnn <- rows %>%
  filter(ensemble_id == 1 & subset == "test") %>%
  group_by(gamma, n_samples, dset, ensemble_id) %>%
  summarise({
    df <- cur_data() # %>% filter(sample < 25)  # less samples = more coverage but less power

    res <- try({
      solve_glm_breslow_for_network(df)
    })
    if(class(res) == "try-error") {
      data.frame()
    }
    else {
      cr <- qt(1-(1-level)/2, n())
      beta_sd <- sqrt(diag(res$covariance))
      beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])

      cov <- (res$betahat - cr * beta_sd <= beta_true) & (beta_true <= res$betahat + cr * beta_sd)
      pow <- !((res$betahat - cr * beta_sd <= 0) & (0 <= res$betahat + cr * beta_sd))
      err <- res$betahat - beta_true

      data.frame(
        coef = 1:length(cov),
        cov = cov, pow = pow, err = err,
        f_cor = (df %>% dplyr::select(c(f, pred_uns)) %>% cor())[1,2],
        method = "dnn"
      )
    }
  }) %>%
  filter(!is.na(cov) & !is.na(pow))


# of the ensemble as a GLMM (our method)
cov_ens_glmm <- rows %>%
  filter(subset == "test") %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    df <- cur_data() # %>% filter(sample < 25)  # less samples = more coverage but less power

    calc <- function(z_type, tau_type) {
      res <- try({
        solve_glm_breslow_for_ensemble(df, z_type, tau_type)
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
        err <- res$betahat - beta_true

        data.frame(
          coef = 1:length(cov),
          cov = cov, pow = pow, err = err,
          f_cor = (df %>% group_by(sample) %>%
                     summarise(ff = mean(f), z = mean(pred_uns)) %>%
                     dplyr::select(-sample) %>% cor())[1,2],
          method = paste("glmm", "z", z_type, "tau", tau_type)
        )
      }
    }

    rbind(
      #calc(z_type = "ens", tau_type = "ens notau"),
      calc(z_type = "ens", tau_type = "ens taupred"),
      calc(z_type = "ens", tau_type = "ens tautrue")
      #calc(z_type = "data", tau_type = "ens notau")

      # makes no sense to use tau when using the true f
      #calc(z_type = "data", tau_type = "ens taupred"),
      #calc(z_type = "data", tau_type = "ens tautrue")
    )
  }) %>%
  filter(!is.na(cov) & !is.na(pow))


# of the ensemble "empirically"
cov_ens_emp <- rows %>%
  group_by(gamma, n_samples, dset) %>%
  summarise({
    dd <- cur_data() %>% filter(sample == 1)

    xx <- lapply(1:4, function(i) {
      make_cov_df(
        n = nrow(dd) - 1,
        beta_true = mean(dd[[paste0("true_beta_", i)]]),
        beta_hat = mean(dd[[paste0("fitted_beta_", i)]]),
        beta_var = var(dd[[paste0("fitted_beta_", i)]]),
        level = level,
        coef = i
      )
    })

    do.call(rbind, xx)
  }) %>%
  mutate(method = "empir", f_cor = NA) %>%
  dplyr::select(-c(beta, est, err, lo, hi))

# # of the ensemble via a GLMM (SLOW!)
# # not sure this approach makes sense because all observations of the same
# # group are the same! that is why we get a shit fit
# df <- rows %>% filter(subset == "test" & dset == 2)
# cov_ens_mixmod <- rows %>%
#   filter(subset == "test") %>%
#   group_by(gamma, n_samples, dset) %>%
#   summarise({
#     df <- cur_data()
#
#     mdl <- try({
#       glmmPQL(
#         y ~ 0 + str_1 + str_2 + str_3 + str_4,
#         random = list(sample = ~ offset(pred_uns)),
#         #y ~ 0 + offset(pred_uns) + str_1 + str_2 + str_3 + str_4,
#         #random = list(sample = ~ 1),
#         family = "poisson", niter = 10,
#         data = df, verbose = F
#       )
#     })
#     summary(mdl)
#     if(any(class(mdl) == "try-error")) {
#       data.frame()
#     }
#     else {
#       cr <- qt(1-(1-level)/2, nrow(df))
#       beta_sd <- sqrt(diag(mdl$varFix))
#       beta_true <- colMeans(df[,grep("true_beta_", colnames(df))])
#       beta_fit <- mdl$coefficients$fixed
#
#       cov <- (beta_fit - cr * beta_sd <= beta_true) & (beta_true <= beta_fit + cr * beta_sd)
#       pow <- !((beta_fit - cr * beta_sd <= 0) & (0 <= beta_fit + cr * beta_sd))
#       err <- beta_fit - beta_true
#
#       data.frame(
#         coef = 1:length(cov),
#         cov = cov, pow = pow, err = err,
#         #f_cor = (df %>% dplyr::select(c(f, pred_uns)) %>% cor())[1,2],
#         method = "mixmod"
#       )
#     }
#   })


lm_data <- rbind(
  cov_dnn %>% filter(ensemble_id == 1),
  cov_ens_glmm, cov_ens_emp #, cov_ens_mixmod
)


# linear model ------------------------------------------------------------


lm_data$method <- factor(
  lm_data$method,
  levels = c(
    "dnn", "empir",
    "glmm z ens tau ens taupred",
    "glmm z ens tau ens tautrue"
  ),
  labels = c("None", "Ensemble", "GLMM (Ours)", "Exact")
)
cmod <- glm(cov ~ gamma + n_samples + method, #+ method:gamma + method:n_samples,
            family = "binomial", data = lm_data)
summary(cmod)
#plot(cmod)

# plots -------------------------------------------------------------------



plot_data <- lm_data %>%
  group_by(method, gamma, n_samples) %>%
  summarise(
    rbind(
      data.frame(
        value = mean(cov),
        cil = binomial_ci(cov, 0.05),
        cih = binomial_ci(cov, 0.95),
        variable = "coverage"
      ),
      data.frame(
        value = mean(pow),
        cil = binomial_ci(pow, 0.05),
        cih = binomial_ci(pow, 0.95),
        variable = "power"
      ),
      data.frame(
        value = mean(err),
        cil = continuous_ci(err, 0.05),
        cih = continuous_ci(err, 0.95),
        variable = "est.err."
      ),
      data.frame(
        value = mean(abs(err)),
        cil = continuous_ci(abs(err), 0.05),
        cih = continuous_ci(abs(err), 0.95),
        variable = "est.abs.err."
      ),
      data.frame(
        value = n(), variable = "n",
        cil = NA, cih = NA
      )
    )
  )

plot_data %>%
  filter(variable %in% c("coverage", "power")) %>%
  filter(method != "ens notau") %>%
  ggplot(aes(x = method, y = value, color = variable)) +
  geom_point(position = position_dodge(0.25), shape = "plus") +
  geom_errorbar(aes(ymin = cil, ymax = cih), position = position_dodge(0.25), width = 0) +
  geom_hline(yintercept = level, colour = "#E69F00") +
  #theme(axis.text.x=element_text(angle=270,hjust=1)) +
  facet_grid(gamma ~ n_samples, labeller = as_labeller(c(
    #`0.25` = "gamma = 0.25", `1` = "gamma = 1.0", `4` = "gamma = 4.0",
    `0` = "tau: 0", `0.5` = "tau: 1/2", `2` = "tau: 2",
    `75` = "Train. size: 75",  `500` = "Train. size: 500", `2000` = "Train. size: 2000"
  ))) +
  theme_bw() +
  labs(x = "", y = "") +
  scale_x_discrete(labels = c("None", "Ensemble", "GLMM (Ours)", "Exact")) +
  scale_color_manual(name = "Variable",  values = c("coverage" = "#E69F00", "power" = "#56B4E9"),
                     labels = c("Coverage", "Power")) +
  theme(axis.text.x=element_text(angle=-45))
ggsave("figures/aa.pdf", width = 6, height = 4)
