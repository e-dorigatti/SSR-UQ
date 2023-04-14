# here we compare the predictions and CIs of linear mixed models against those of
# an ensemble and individual networks


library(ggplot2)
library(nlme)
library(GGally)
library(gridExtra)
library(dplyr)
library(keras)
library(reshape2)

get_dataset <- function(count, include_images = TRUE, noise_sd = 0.25, unstructured_coef = 1) {
  # the prediction task is to compute a+x with a and x integers from zero to nine.
  # x is a digit image from mnist and a is one-hot encoded
  # when include_images is false, the dataset has no image data

  aa <- array(0, dim=c(count, 10))
  aai <- array(0, dim=c(count))
  xxi <- array(0, dim=c(count))
  yy <- array(0, dim=c(count))
  if(include_images) {
    mnist <- dataset_mnist()
    xx <- array(0, dim=c(count, 28, 28, 1))
  }
  else {
    xx <- array(0, dim=c(0, 28, 28, 1))
  }

  for(i in 1:count) {
    xd <- sample.int(10, 1)
    ad <- sample.int(10, 1)

    x <- (xd - 5.5) * unstructured_coef
    a <- ad - 5.5

    aa[i, ad] <- 1
    aai[i] <- a
    xxi[i] <- x
    yy[i] <- a + x

    if(include_images) {
      digit_idx <- (1:nrow(mnist$train$x))[mnist$train$y == (xd - 1)]
      image_idx <- sample(digit_idx, 1)
      xx[i,,,1] <- mnist$train$x[image_idx,,] / 255.0
    }
  }

  if(!is.null(noise_sd)) {
    yy <- yy + rnorm(count, sd = noise_sd)
  }

  list(
    aa = aa, aai = aai,
    xx = xx, xxi = xxi,
    yy = yy
  )
}


get_new_sddr <- function() {
  l2 <- 1e-5
  digit_encoder <- keras_model_sequential(name = "digit_encoder") %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same", kernel_regularizer = regularizer_l2(l2)) %>%
    layer_activation_leaky_relu() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", kernel_regularizer = regularizer_l2(l2)) %>%
    layer_activation_leaky_relu() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same", kernel_regularizer = regularizer_l2(l2)) %>%
    layer_activation_leaky_relu() %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(64) %>%
    layer_dense(1, use_bias = FALSE)

  input_a <- layer_input(shape = c(10))
  input_x <- layer_input(shape = c(28, 28, 1))

  # network output does not get its own coefficient but is simply added
  # to the predictions of a linear model on structured features
  model_output <- layer_add(list(
    input_x %>% digit_encoder,
    input_a %>% layer_dense(1, use_bias = FALSE)
  ))

  model <- keras_model(
    inputs = list(input_a, input_x),
    outputs = model_output
  )

  list(
    model = model,
    digit_encoder = digit_encoder
  )
}


get_trained_sddr <- function(train_dset, val_dset, epochs = 500, batch_size = 128, verbose = 2, lrate = 1e-2) {
  # builds a sddr model and trains it on a random dataset of the given size
  # returns history, model, data, and last-layer features

  sddr <- get_new_sddr()
  sddr$model %>% compile(loss = "mse", optimizer = optimizer_adam(lrate), metrics = c("mae"))

  hist <- sddr$model %>% fit(
    list(train_dset$aa, train_dset$xx),
    train_dset$yy,
    validation_data=list(list(val_dset$aa, val_dset$xx), val_dset$yy),
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose,
    callbacks = list(
      callback_early_stopping(patience = 8, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(patience = 5)
    )
  )

  list(
    history = hist,
    sddr = sddr,
    train_dset = train_dset,
    val_dset = val_dset
  )
}


get_last_layer_features_dataset <- function(sddr, dataset) {
  # returns a new dataset containing the structured features,
  # the predictions of the network and the response variable

  preds <- sddr$digit_encoder %>% predict(dataset$xx)
  df <- as.data.frame(cbind(dataset$aai, dataset$xxi, preds, dataset$yy))
  colnames(df) <- c("a", "xi", "x.1", "y")
  df$sample <- 1:nrow(df)

  df
}


# # quick test
# tds <- get_dataset(500)
# vds <- get_dataset(500)
# eds <- get_dataset(500)
# s <- get_trained_sddr(tds, vds, epochs = 20)
# l <- get_last_layer_features_dataset(s$sddr, tds)
# head(l)


train_ensemble_on_dataset <- function(ensemble_size, train_dset, val_dset, test_dset) {
  rows <- c()
  for(ens_rep in 1:ensemble_size) {
    e <- get_trained_sddr(train_dset, val_dset)

    # compute last layer features
    llf_train <- get_last_layer_features_dataset(e$sddr, train_dset)
    llf_val <- get_last_layer_features_dataset(e$sddr, val_dset)
    llf_test <- get_last_layer_features_dataset(e$sddr, test_dset)

    llf_train$subset <- "train"
    llf_val$subset <- "val"
    llf_test$subset <- "test"

    # add metadata
    llf <- rbind(llf_train, llf_val, llf_test)
    llf$train_size <- nrow(train_dset$y)
    llf$ensemble_id <- ens_rep

    rows = rbind(rows, llf)
  }

  rows
}

run_simulation <- function() {
  # repatedly trains SDDR models for increasingly large training set sizes
  # for each repetition and train set size, returns last-layer features of the entire dataset

  fname <- "uq-mixed-model.RDS"
  if(file.exists(fname)){
    return(readRDS(fname))
  }

  dset_id <- 0
  rows <- c()

  for(uc in c(0, 0.5, 2.0)) {
    # large common test set for faithful evaluation
    test_dset <- get_dataset(1000, unstructured_coef = uc)

    for(samples in c(250, 2500)) {
      for(dset_rep in 1:25) {
        dset_id <- dset_id + 1
        train_dset <- get_dataset(samples, unstructured_coef = uc)
        val_dset <- get_dataset(250, unstructured_coef = uc)

        rs <- train_ensemble_on_dataset(25, train_dset, val_dset, test_dset)
        rs$dataset <- dset_id
        rs$dataset_rep <- dset_rep
        rs$unstructured_coef <- uc

        rows = rbind(rows, rs)

        saveRDS(rows, file = fname)
      }
    }
  }

  return(rows)
}

#rows <- run_simulation()  # full dataset (pretty large)
#rows <- readRDS("uq-mixed-model-small.RDS")  # small subset
#rows <- readRDS("results/uq-mixed-model.RDS")

rows <- readRDS("results/uq-mixed-model-big.RDS")



# column groups
key_columns <- c("subset", "train_size", "ensemble_id", "dataset", "unstructured_coef")
coef_names_struc <- c("a.0", "a.1", "a.2", "a.3", "a.4", "a.5", "a.6", "a.7", "a.8", "a.9")
coef_names_uns <- c("x.1")
coef_names_all <- c(coef_names_struc, coef_names_uns)

onehot_dataset <- function(x) {
  mat <- model.matrix(~ 0 + as.factor(x$a))
  colnames(mat) <- coef_names_struc
  as.data.frame(cbind(x, mat))
}

compute_estimation_error <- function(dataset) {
  sm_all <- summary(lm(
    as.formula(paste("y ~ 0 +", paste(coef_names_all, collapse = " + "))),
    dataset
  ))
  res <- data.frame(
    error = sm_all$coefficients[coef_names_struc,"Estimate"]
  )

  # compute difference with true coefficients
  res <- res - seq(-4.5, 4.5)

  # add fake coefficient with mean abs error of the models
  res["resid", "error"] <- mean(abs(sm_all$residuals))

  res$coef <- rownames(res)

  res
}

esterr <- rows %>%
  group_by(unstructured_coef, dataset, train_size, ensemble_id, subset) %>%
  group_modify(~ {
    compute_estimation_error(onehot_dataset(.x))
  })

esterr_summ <- esterr %>% group_by(unstructured_coef, train_size, subset, coef) %>%
  summarise(mean_error = mean(error), se_error = sd(error) / sqrt(length(error)))

saveRDS(esterr_summ, "results/esterr_summ.RDS")


esterr_summ %>%
  filter(unstructured_coef %in% c(0, 0.5, 2) & coef != "resid") %>%
  mutate(
    Coefficient = sub("a.", "", coef),
    subset = factor(subset, levels = c("train", "val", "test"),
                    labels = c("Train.", "Valid.", "Test"))
  ) %>%
  rename(
    tau = unstructured_coef,
    `Train. size` = train_size,
    Subset = subset
  ) %>%
  ggplot(aes(x = Coefficient, y = mean_error, group = Coefficient)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean_error - 2 * se_error, ymax = mean_error + 2 * se_error)) +
  geom_hline(yintercept = 0.0) +
  facet_grid(rows = vars(tau), cols = vars(Subset, `Train. size`), labeller = label_both) +
  labs(y = "Estimation Error") +
  theme_bw()
ggsave("results/coef-bias.pdf", width = 12, height = 6)

