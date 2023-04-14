Sys.setenv(OMP_NUM_THREADS=1)


library(ggplot2)
library(dplyr)
library(reshape2)
library(R6)
library(parallel)


PoissonSddrSimulator <- R6Class(
  "PoissonSddrSimulator",
  public = list(
    dim_str = 4,
    dim_uns = 8,

    # dnn used for the data generating process
    dataset_n_hidden_layers = 1,
    dataset_layer_size = 8,
    dataset_activation = "tanh",

    # dnn used to fit the dataset
    dnn_n_hidden_layers = 3,
    dnn_layer_size = 12,
    dnn_weight_decay = 1e-5,
    dnn_activation = "relu",

    # training params
    epochs = 75,
    batch_size = 128,
    verbose = 0,
    lrate = 0.075,
    ensemble_size = 25,

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
        model <- model %>% layer_dense(1, input_shape = din)
      }

      model
    },

    make_dataset = function(n_samples, beta, dnn, gamma = 1) {
      stopifnot(length(beta) == self$dim_str)
      x_struc <- matrix(rnorm(n_samples * self$dim_str), nrow = n_samples)
      x_uns <- matrix(rnorm(n_samples * self$dim_uns), nrow = n_samples)

      t_struc <- (x_struc %*% beta)[,1]
      t_uns <- predict(dnn, x_uns)[,1]
      t_uns <- t_uns / sd(t_uns)

      t_eta <- t_struc + gamma * t_uns
      obs <- rpois(n_samples, exp(t_eta))

      mask <- (t_eta > -3) & (t_eta < 3)
      list(
        beta = beta,
        x_struc = x_struc[mask,],
        x_uns = x_uns[mask,],
        t_struc = t_struc[mask],
        t_uns = gamma * t_uns[mask],
        t_eta = t_eta[mask],
        y = obs[mask]
      )
    },

    get_dataset = function(n_train, n_val, n_test, gamma) {
      beta <- rnorm(self$dim_str, sd = 1 / sqrt(self$dim_str))
      dnn <- self$get_dnn(self$dim_uns, self$dataset_n_hidden_layers,
                          self$dataset_layer_size, self$dataset_activation)

      list(
        train = self$make_dataset(n_train, beta, dnn, gamma),
        val = self$make_dataset(n_val, beta, dnn, gamma),
        test = self$make_dataset(n_test, beta, dnn, gamma)
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
      )) %>% activation_exponential()

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

    get_trained_sddr = function(train_dset, val_dset, pretrain_structured = FALSE) {  # FIXME should be true
      # builds a sddr model and trains it on a random dataset of the given size
      # returns history, model, data, and last-layer features

      sddr <- self$get_new_sddr()

      if(pretrain_structured) {
        coefs <- lm(train_dset$y ~ 0 + train_dset$x_struc)$coefficients
        sddr$structured_head$set_weights(list(as.matrix(coefs)))
      }

      sddr$model %>% compile(
        loss = loss_poisson, optimizer = optimizer_adam(self$lrate), metrics = c("mae"))

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
        dataset$t_uns
      ))
      colnames(df) <- c(
        paste0("str_", 1:self$dim_str),
        paste0("uns_", 1:self$dim_uns),
        "pred_uns", "preds",
        "eta", "y", "f"
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


settings <- expand.grid(
  1:150,
  c(75, 500, 2000),
  c(0, 0.5, 2)
)


rows <- do.call(rbind, mclapply(1:nrow(settings), function(i) {
  # make tensorflow use only a single core
  Sys.setenv(OMP_NUM_THREADS=1)
  tf <- reticulate::import("tensorflow", convert = TRUE, delay_load = FALSE)
  tf$config$threading$set_inter_op_parallelism_threads(1L)
  tf$config$threading$set_intra_op_parallelism_threads(1L)
  library(keras)

  d <- settings[i, 1]
  n <- settings[i, 2]
  g <- settings[i, 3]
  cat("doing", i, d, n, g, "\n")

  sddr_sim <- PoissonSddrSimulator$new()
  dset <- sddr_sim$get_dataset(n, 100, 100, gamma = g)
  res <- sddr_sim$train_ensemble_on_dataset(dset)

  # add true beta
  res[,paste0("true_beta_", 1:sddr_sim$dim_str)] <- matrix(
    rep(dset$train$beta, nrow(res)), ncol = sddr_sim$dim_str, byrow = T)

  res <- res %>% filter(subset != "train")
  res$dset <- i
  res$n_samples <- n
  res$gamma <- g

  res
}, mc.cores = 6))

saveRDS(rows, "poisson-dnn.RDS")



