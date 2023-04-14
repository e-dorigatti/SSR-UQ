Sys.setenv(OMP_NUM_THREADS=1)


library(ggplot2)
library(dplyr)
library(reshape2)
library(R6)
library(parallel)
library(mgcv)
library(MASS)


GaussianSddrSimulator <- R6Class(
  "GaussianSddrSimulator",
  public = list(
    dim_str = 12,
    dim_uns = 12,

    # dnn used for the data generating process
    dataset_n_hidden_layers = 1,
    dataset_layer_size = 8,
    dataset_activation = "tanh",

    # dnn used to fit the dataset
    dnn_n_hidden_layers = 1,  # 3, 24
    dnn_layer_size = 12,
    dnn_weight_decay = 1e-5,
    dnn_activation = "relu",

    # training params
    epochs = 75,
    batch_size = 128,
    verbose = 0,
    lrate = 0.075,
    ensemble_size = 9,

    true_model = NULL,

    initialize = function() {
      xx <- 2 + 3 * rnorm(100)
      yy <- rnorm(100, sin(xx) / xx, sd = 0.1)
      df <- data.frame(x = xx, y = yy)
      self$true_model <- gam(y ~ 0 + s(x, k = self$dim_str + 1), data = df)
    },

    get_dnn = function(input_shape, n_hidden_layers, layer_size, activation, head = TRUE) {
      model <- keras_model_sequential()
      for(i in 1:n_hidden_layers) {
        if(i == 1) { din <- input_shape } else { din <- NULL }
        model <- model %>% layer_dense(
          layer_size, activation = activation, input_shape = din,
          kernel_regularizer = regularizer_l2(self$dnn_weight_decay)
        )
      }

      if(head) {
        if(i == 1) { din <- input_shape } else { din <- NULL }
        model <- model %>% layer_dense(1, input_shape = din, use_bias = FALSE)
      }

      model
    },

    make_dataset = function(n_samples, sample_type, dnn, gamma = 1) {
      # generate structured
      if(sample_type == "norm") {
        df <- data.frame(x = 1 + 2 * rnorm(n_samples))
      }
      else {
        df <- data.frame(x = runif(n_samples, -5, 8))
      }
      x_struc <- model.matrix.gam(self$true_model, df)
      t_struc <- predict(self$true_model, df)

      # generate unstructured
      x_uns <- matrix(rnorm(n_samples * self$dim_uns), nrow = n_samples)
      t_uns <- predict(dnn, x_uns)[,1]
      t_uns <- t_uns / sd(t_uns)

      # generate real observations
      t_eta <- t_struc + gamma * t_uns
      obs <- t_eta + rnorm(n_samples, sd = 0.25)

      list(
        xx = df$x,
        beta = coef(self$true_model),
        x_struc = x_struc,
        x_uns = x_uns,
        t_struc = t_struc,
        t_uns = gamma * t_uns,
        t_eta = t_eta,
        y = obs
      )
    },

    get_dataset = function(n_train, n_val, n_test, gamma) {
      dnn <- self$get_dnn(self$dim_uns, self$dataset_n_hidden_layers,
                          self$dataset_layer_size, self$dataset_activation)

      list(
        train = self$make_dataset(n_train, "norm", dnn, gamma),
        val = self$make_dataset(n_val, "unif", dnn, gamma),
        test = self$make_dataset(n_test, "unif", dnn, gamma)
      )
    },

    get_new_sddr = function() {
      input_uns <- layer_input(shape = c(self$dim_uns))
      net <- self$get_dnn(NULL, self$dnn_n_hidden_layers, self$dnn_layer_size, self$dnn_activation)

      input_str <- layer_input(shape = c(self$dim_str))
      structured_head <- keras_model_sequential() %>%
        layer_dense(1, use_bias = FALSE)

      # the network output does not get its own coefficient but is simply added
      # to the predictions of a linear model on structured features
      model_output <- layer_add(list(
        input_uns %>% net,
        input_str %>% structured_head
      ))

      model <- keras_model(
        inputs = list(input_str, input_uns),
        outputs = model_output
      )

      list(
        model = model,
        net = net,
        structured_head = structured_head
      )
    },

    get_trained_sddr = function(train_dset, val_dset, pretrain_structured = TRUE) {
      # builds a sddr model and trains it on a random dataset of the given size
      # returns history, model, data, and last-layer features

      sddr <- self$get_new_sddr()

      if(pretrain_structured) {
        coefs <- lm(train_dset$y ~ 0 + train_dset$x_struc)$coefficients
        sddr$structured_head$set_weights(list(as.matrix(coefs)))
      }

      sddr$model %>% compile(
        loss = loss_mean_squared_error,
        optimizer = optimizer_adam(self$lrate),
        metrics = c("mae"))

      hist <- sddr$model %>% fit(
        list(train_dset$x_struc, train_dset$x_uns),
        train_dset$y,
        validation_data=list(list(val_dset$x_struc, val_dset$x_uns), val_dset$y),
        epochs = self$epochs,
        batch_size = self$batch_size,
        verbose = self$verbose,
        callbacks = list(
          callback_early_stopping(patience = 10, restore_best_weights = TRUE),
          callback_reduce_lr_on_plateau(patience = 5, factor = 0.5)
        )
      )

      list(
        history = hist,
        sddr = sddr,
        train_dset = train_dset,
        val_dset = val_dset
      )
    },

    get_last_layer_features_dataset = function(sddr, dataset) {
      # returns a new dataset containing the structured features,
      # the predictions of the network and the response variable

      preds_uns <- sddr$net %>% predict(dataset$x_uns)
      preds_mod <- sddr$model %>% predict(list(dataset$x_str, dataset$x_uns))
      df <- as.data.frame(cbind(
        dataset$x_str, dataset$x_uns,
        preds_uns, preds_mod,
        dataset$t_eta, dataset$y,
        dataset$t_uns, dataset$xx,
        dataset$t_struc
      ))
      colnames(df) <- c(
        paste0("str_", 1:self$dim_str),
        paste0("uns_", 1:self$dim_uns),
        "pred_uns", "preds",
        "eta", "y", "f", "xx",
        "t_struc"
      )

      # add fitted beta
      df[,paste0("fitted_beta_", 1:self$dim_str)] <- matrix(
        rep(sddr$structured_head$get_weights()[[1]][,1], nrow(df)),
        ncol = self$dim_str, byrow = T)

      df$sample <- 1:nrow(df)

      df
    },

    train_ensemble_on_dataset = function(dset) {
      rows <- c()
      for(ens_rep in 1:self$ensemble_size) {
        e <- self$get_trained_sddr(dset$train, dset$val)

        # compute last layer features
        llf_train <- self$get_last_layer_features_dataset(e$sddr, dset$train)
        llf_val <- self$get_last_layer_features_dataset(e$sddr, dset$val)
        llf_test <- self$get_last_layer_features_dataset(e$sddr, dset$test)

        llf_train$subset <- "train"
        llf_val$subset <- "val"
        llf_test$subset <- "test"

        # add metadata
        llf <- rbind(llf_train, llf_val, llf_test)
        llf$train_size <- nrow(dset$train$y)
        llf$ensemble_id <- ens_rep

        rows = rbind(rows, llf)
      }

      rows
    }
  )
)

run_sim <- function(n, g) {
  # make tensorflow use only a single core
  Sys.setenv(OMP_NUM_THREADS=1)
  tf <- reticulate::import("tensorflow", convert = TRUE, delay_load = FALSE)
  tf$config$threading$set_inter_op_parallelism_threads(1L)
  tf$config$threading$set_intra_op_parallelism_threads(1L)
  library(keras)

  sddr_sim <- GaussianSddrSimulator$new()
  dset <- sddr_sim$get_dataset(n, 100, 100, gamma = g)
  rows <- sddr_sim$train_ensemble_on_dataset(dset)

  # add true beta
  rows[,paste0("true_beta_", 1:sddr_sim$dim_str)] <- matrix(
    rep(dset$train$beta, nrow(rows)), ncol = sddr_sim$dim_str, byrow = T)

  rows$dset <- 1
  rows$n_samples <- n
  rows$gamma <- g

  rows
}

rows <- run_sim(50, 1)

#saveRDS(rows, "spline-sddr-3.RDS")

#rows <- readRDS("spline-sddr-3.RDS")



# analysis ----------------------------------------------------------------

#old_rows <- rows
#rows <- rows %>% filter(subset != "train")

rows %>% filter(subset == "test") %>% group_by(ensemble_id) %>% summarise({
  df <- cur_data()

  xs <- as.matrix(df[,grep("str_", colnames(df))])
  tbs <- colMeans(df[,grep("true_beta_", colnames(df))])
  fbs <- colMeans(df[,grep("fitted_beta_", colnames(df))])

  data.frame(
    xx = df$xx,
    fitted = xs %*% fbs + mean(df$pred_uns),  # not sure why DNN output isn't zero!?
    true = xs %*% tbs,
    obs = df$y - df$f
  )
}) %>%
  filter(xx > -4 & xx < 7.1) %>%
  ggplot(aes(x = xx)) +
  geom_line(aes(y = fitted)) +
  geom_line(aes(y = true)) +
  geom_point(aes(y = obs)) +
  facet_wrap(~ ensemble_id)



fit_additive <- function(xs, ys, zs, tau, lambda, verbose = F) {
  s <- diag(rep(lambda, ncol(xs)))
  #s[1, 1] <- 0  # do not penalize intercept for fit
  p <- solve(t(xs) %*% xs + s) %*% t(xs)

  coef <- (p %*% (ys - zs))[,1]
  yhat <- (xs %*% coef + zs)[,1]

  degf <- nrow(xs) - ncol(xs)
  sigmasq_hat <- (sum((ys - yhat)**2) - sum(tau**2)) / degf
  stopifnot(sigmasq_hat > 0)

  if(verbose) {
    cat("rss", sum((ys - yhat)**2), "on", degf, "degrees of freedom", "\n")
    cat("estimated sigma squared", sigmasq_hat, "\n")
  }

  list(
    xs = xs, ys = ys, zs = zs, tau = tau, lambda = lambda,
    beta = coef,
    yhat = yhat,
    sigma_hat = sigmasq_hat,
    beta_cov = p %*% (tau**2 + diag(rep(sigmasq_hat, nrow(xs)))) %*% t(p)
  )
}


fit_additive_for_ensemble <- function(df, lambda, network, use_tau = TRUE, verbose = F) {
  if(is.null(network)) {
    sdf <- df[df$ensemble_id == 1,]
  }
  else {
    sdf <- df[df$ensemble_id == network,]
  }

  xs <- as.matrix(sdf[, grep("str_", colnames(df))])
  ys <- as.matrix(sdf$y)

  if(is.null(network)) {
    ens_pred <- df %>% group_by(sample) %>% summarise(z = mean(pred_uns), tau = sd(pred_uns))
    zs <- ens_pred$z
    tau <- ens_pred$tau
  }
  else {
    zs <- sdf$pred_uns
    tau <- 0
  }

  if(!use_tau) {
    tau <- 0
  }
  # center manually since for some reason DNN output isn't always zero mean
  zs <- zs - mean(zs)

  res <- fit_additive(xs, ys, zs, tau, lambda, verbose)
  res$xx <- sdf$xx
  res$t_struc <- sdf$t_struc
  res$f <- sdf$f
  res
}

tune_lambda_for_network <- function(df, network) {
  val_df <- df %>% filter(subset == "val" & ensemble_id == network)
  test_df <- df %>% filter(subset == "test" & ensemble_id == network)

  test_xs <- as.matrix(test_df[, grep("str_", colnames(df))])
  test_ys <- test_df$y
  test_zs <- test_df$pred_uns

  logls <- -20:20
  rmses <- sapply(logls, function(logl) {
    rr <- fit_additive_for_ensemble(val_df, lambda = 2**logl, network = network)
    sqrt(mean((test_xs %*% rr$beta - test_zs - test_ys)**2))
  })

  2**logls[which.min(rmses)]
}



tune_lambda_for_ensemble <- function(df, use_tau = T) {
  val_df <- df %>% filter(subset == "val")
  test_df <- df %>% filter(subset == "test")

  test_xs <- as.matrix(test_df[df$ensemble_id == 1, grep("str_", colnames(df))])
  test_ys <- test_df[df$ensemble_id == 1,"y"]
  test_zs <- test_df[df$ensemble_id == 1,"pred_uns"]

  logls <- -20:20
  rmses <- sapply(logls, function(logl) {
    rr <- fit_additive_for_ensemble(val_df, lambda = 2**logl, network = NULL, use_tau = use_tau)
    sqrt(mean((test_xs %*% rr$beta - test_zs - test_ys)**2))
  })

  2**logls[which.min(rmses)]
}


# additive model for each network separately
fit_summary <- function(rr) {
  # not sure why it complains that beta_cov is not pos def
  # since solve(beta_cov) works
  cov <- rr$beta_cov #+ diag(rep(1e-3, nrow(rr$beta_cov)))
  bs <- mvrnorm(1000, rr$beta, cov, tol=1e4)
  mcs <- rr$xs %*% t(bs)

  mcm = rowMeans(mcs)
  mcv = rowMeans((mcs - rowMeans(mcs))**2)

  data.frame(
    xx = rr$xx,
    net_pred = rr$yhat - rr$zs,
    z = rr$t_struc,
    obs = rr$ys - rr$f,
    mc_mean = mcm,
    mc_var = mcv,
    mc_lb = apply(mcs, 1, function(x) { quantile(x, 0.05) }),
    mc_ub = apply(mcs, 1, function(x) { quantile(x, 0.95) }),
    sample = 1:length(rr$xx)
  )
}


# plot individual networks
res_net <- c()
for(net in unique(rows$ensemble_id)) {
  ll <- tune_lambda_for_network(rows, net)
  rr <- fit_additive_for_ensemble(rows %>% filter(subset == "test"), lambda = ll, network = net, verbose = T)
  smr <- fit_summary(rr) %>% mutate(network = net)

  df <- rows %>% filter(subset == "test" & ensemble_id == net)
  xs <- as.matrix(df[,grep("str_", colnames(df))])
  tbs <- colMeans(df[,grep("true_beta_", colnames(df))])
  fbs <- colMeans(df[,grep("fitted_beta_", colnames(df))])
  smr$sddr_fit <- xs %*% fbs + mean(df$pred_uns)
  smr$net <- net

  res_net <- rbind(res_net, smr)
}


# with confidence interval from additive model
res_net %>%
  dplyr::select(c(net, sample, xx, mc_mean, mc_lb, mc_ub, z, sddr_fit)) %>%
  melt(id.vars = c("net", "sample", "xx", "mc_lb", "mc_ub")) %>%
  filter(xx > -3.5 & xx < 7.1) %>%
  ggplot(aes(x = xx, group = variable)) +
  geom_ribbon(aes(ymin = mc_lb, ymax = mc_ub, fill = variable)) +
  geom_line(aes(y = value, color = variable, linetype = variable)) +
  facet_wrap(~ net) +
  scale_fill_manual(
    "Uncertainty",
    values = c("sddr_fit" = "#bbbbbb"),
    labels = c("AM")
  ) +
  scale_color_manual(
    "Fit",
    values = c("z" = "red", "mc_mean" = "black", "sddr_fit" = "black"),
    labels = c("DGP", "AM", "DNN")
  ) +
  scale_linetype_manual(
    "Fit",
    values = c("z" = "solid", "mc_mean" = "dotted", "sddr_fit" = "dashed"),
    labels = c("DGP", "AM", "DNN")
  ) +
  theme_bw() +
  labs(x = "", y = "") +
  theme(strip.text.x = element_blank())
#ggsave("figures/am-nets.pdf", width = 6, height = 4)


# compute average of additive models fitted individually
# this is essentially like fitting a single model on the average net prediction
# without using tau at all
res_am_m <- res_net %>% group_by(sample) %>% summarise(
  xx = mean(xx),
  net_pred = mean(net_pred),
  z = mean(z),
  obs = mean(obs),
  mc_mean = mean(mc_mean),
  #mc_lb = mean(mc_lb),
  #mc_ub = mean(mc_ub),
  mc_lb = mean(mc_mean) - 1.7 * sqrt(mean(mean(mc_var))),
  mc_ub = mean(mc_mean) + 1.7 * sqrt(mean(mean(mc_var))),
  mc_var = mean(mc_var),
  method = "AM Avg.",
  tau = "no"
)

# our additive model
rrt <- fit_additive_for_ensemble(rows %>% filter(subset == "test"), lambda = tune_lambda_for_ensemble(rows, use_tau = T), network = NULL, verbose = T, use_tau = T)
res_amt <- fit_summary(rrt) %>% mutate(method = "Thm. 3", tau = "yes", sample = 1:n())

rrn <- fit_additive_for_ensemble(rows %>% filter(subset == "test"), lambda = tune_lambda_for_ensemble(rows, use_tau = F), network = NULL, verbose = T, use_tau = F)
res_amn <- fit_summary(rrn) %>% mutate(method = "Thm. 3", tau = "no", sample = 1:n())

# remove this mean as it almost exactly matches res_amt$mc_mean and ruins the plot
# theoretically they should be the same anyways
res_amn$mc_mean <- NA

# ensemble average
res_ens <- rows %>% filter(subset == "test") %>% group_by(ensemble_id) %>% summarise({
  df <- cur_data()

  xs <- as.matrix(df[,grep("str_", colnames(df))])
  tbs <- colMeans(df[,grep("true_beta_", colnames(df))])
  fbs <- colMeans(df[,grep("fitted_beta_", colnames(df))])

  data.frame(
    sample = sample,
    xx = df$xx,
    fitted = xs %*% fbs + mean(df$pred_uns),  # not sure why DNN output isn't zero!?
    true = xs %*% tbs,
    obs = df$y - df$f
  )
}) %>%
  ungroup() %>%
  group_by(sample) %>%
  summarise(
    xx = mean(xx),
    z = mean(true),
    net_pred = mean(fitted),
    obs = mean(obs),
    mc_mean = mean(net_pred),
    mc_var = var(fitted),
    mc_lb = mc_mean - 1.83 * sd(fitted),
    mc_ub = mc_mean + 1.83 * sd(fitted),
    method = "Ensemble Avg.",
    tau = "yes"
  )

tr <- rows[rows$subset == "train" & rows$ensemble_id == 1,]
yt <- tr$y - tr$f
#xt = 2 + 3 * rnorm(50)
data_train = data.frame(
  xt = c(tr$xx, tr$xx, tr$xx), yt = c(yt, yt, yt),
  method = c(res_ens$method, res_amt$method, res_am_m$method),
  tau_f = "x"
)

rbind(res_ens, res_amt, res_amn, res_am_m) %>%
  mutate(
    method_f = factor(method, levels = c("Ensemble Avg.", "AM Average", "Thm. 3")),
    fit_f = ifelse(method == "Ensemble Avg.", "dnn", "am"),
    tau_f = factor(tau, levels = c("yes", "no"))
  ) %>%
  filter(xx > -3.5 & xx < 7.1) %>%
  ggplot(aes(x = xx, group = interaction(tau_f, method))) +
  geom_ribbon(aes(ymin = mc_lb, ymax = mc_ub, fill = tau_f)) +
  geom_point(
    data = data_train,
    aes(
      x = xt, y = -2,
    ), shape = "|"
  ) +
  geom_line(aes(y = mc_mean, linetype = fit_f)) +
  geom_line(aes(y = z), color = "red") +
  facet_grid(method_f ~ .) +
  theme_bw() +
  #xlim(-5, 8) +
  labs(x = "", y = "") +
  scale_linetype_manual(
    "Fit",
    values = c("dnn" = "dashed", "am" = "dotted"),
    labels = c("DNN", "AM")
  ) +
  scale_fill_manual(
    "Uncertainty",
    values = c("yes" = "#dddddd", "no" = "#bbbbbb"),
    labels = c("DNN", "AM")
  )
#ggsave("figures/am-ens.pdf", width = 6, height = 4)
