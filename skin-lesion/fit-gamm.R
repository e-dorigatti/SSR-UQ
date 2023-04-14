library(mgcv)
library(Matrix)
library(parallel)
library(tidyr)


# penalized Fisher scoring for logistic reg
solvePenGLM <- function(design_matrix, response_vector,
                        penalty_matrix, offset, epsilon=.0001,
                        lr = 0.5) {

  X <- as(design_matrix, "diagonalMatrix")
  y <- as(response_vector, "Matrix")
  P <- as(penalty_matrix, "diagonalMatrix")
  offset <- as.matrix(offset)

  # calc functions
  calc_add_pred <- function(X, beta, offset) plogis(as.vector(crossprod(X, beta) + offset))
  calc_weights <- function(eta) as(diag(eta * (1-eta)), "diagonalMatrix")
  calc_I <- function(X, W, P) crossprod(X %*% W, X) + P
  calc_U <- function(X, y, eta, P, beta) crossprod(X, y - eta) - crossprod(P, beta)
  # calc iter
  calc_iter <- function(X, beta, offset, y, P){
    eta <- calc_add_pred(X, as(beta, "Matrix"), offset)
    W <- calc_weights(eta)
    I <- calc_I(X, W, P)
    U <- calc_U(X, y, eta, P, as(beta, "Matrix"))
    return(list(I = I, U = U))
  }

  # initialize variables for iteration =>
  beta_old <- matrix(rep(0, ncol(X)), nrow=ncol(X), ncol=1, byrow=FALSE, dimnames=NULL)
  derivs <- calc_iter(X, beta_old, offset, y, P)
  iter_I <- derivs$I
  iter_U <- derivs$U
  fisher_scoring_iterations  <- 0

  # iterate until difference between abs(beta_new - beta_old) < epsilon =>
  while(TRUE) {

    # Fisher Scoring Update Step =>
    fisher_scoring_iterations <- fisher_scoring_iterations + 1
    cat("--> Fisher Iteration", fisher_scoring_iterations, " - ")
    beta_new <- beta_old + lr * solve(iter_I, iter_U)

    absdiff <- abs(beta_new - beta_old)

    cat("Max difference in betas", max(absdiff), "\n")

    if (all(absdiff < epsilon)) {
      model_parameters  <- beta_new
      fitted_values     <- calc_add_pred(X, model_parameters, offset)
      covariance_matrix <- solve(iter_I)
      break
    } else {
      derivs <- calc_iter(X, beta_new, offset, y, P)
      iter_I <- derivs$I
      iter_U <- derivs$U
      beta_old <- beta_new
    }

  } # end while

  summaryList <- list(
    'model_parameters'=model_parameters,
    'covariance_matrix'=covariance_matrix,
    'fitted_values'=fitted_values,
    'number_iterations'=fisher_scoring_iterations
  )
  return(summaryList)
}

fit_gamm <- function(df) {
  fml = target ~ site + sex + s(age_approx,by=sex) + s(patient_id, bs = "re") + offset(ens_pred)
  mdl <- gam(fml, family = "binomial", data = df)
  pred_gam_old <- pred_gam_new <- predict(mdl)
  iter <- 0

  while(TRUE){

    iter <- iter + 1
    cat("Iteration ", iter, "\n")
    df$dummy_offset <- pred_gam_old <- pred_gam_new
    res <- solvePenGLM(design_matrix = diag(rep(1,nrow(df))),
                       response_vector = df$target,
                       penalty_matrix = diag(1/df$ens_var),
                       offset = df$dummy_offset)
    df$ofs <- as.vector(res$model_parameters) + df$ens_pred
    fml_off = target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re") + offset(ofs)
    cat("Gam fitting ...")
    mdl <- gam(fml_off, family = "binomial", data = df)
    pred_gam_new <- predict(mdl)
    if(all(abs(pred_gam_new - pred_gam_old) < 0.0001)){
      break
    }else{
      cat("Max Difference", max(abs(pred_gam_new - pred_gam_old)), "\n")
    }
  }

  mdl
}


do_penalty <- function(pen) {
  preds <- all_preds[all_preds$penalty == pen,]
  cat(nrow(preds), "observations\n")

  df <- readRDS("data/train-processed.RDS")[preds$idx + 1,]
  preds$outcome <- df$target
  nets <- preds[,grep("network_", colnames(preds))]
  nets <- nets[,colSums(is.na(nets)) < nrow(nets)]

  df$ens_pred <- rowMeans(nets)
  df$ens_var <- rowMeans((nets - df$ens_pred)**2)

  cat("fitting on tune\n")
  mdl <- fit_gamm(df[df$subset == "tune",])
  saveRDS(mdl, paste0("mixmod_tune_pen=", pen, ".RDS"))

  cat("fitting on test\n")
  mdl <- fit_gamm(df[df$subset == "test",])
  saveRDS(mdl, paste0("mixmod_test_pen=", pen, ".RDS"))
}

all_preds <- read.csv2("ensemble_preds_heldout_nost.csv", sep = ",", dec = ".")
cat("got penalties", unique(all_preds$penalty), "\n")

#parallel::mclapply(unique(all_preds$penalty), do_penalty, mc.cores = 20)

args <- commandArgs(trailingOnly = TRUE)
if(length(args) > 0) {
    pen <- as.double(args[1])
    cat("doing penalty", pen, "\n")
    do_penalty(pen)
}



# preds <- read.csv2("nostruc/ensemble-preds-nostruc.csv", sep = ",", dec = ".")
# df <- readRDS("data/train-processed.RDS")[preds$idx + 1,]
# df$patient_id <- as.factor(df$patient_id)
# preds$outcome <- df$target
# summary(preds)
# nets <- preds[,grep("network_", colnames(preds))]
# df$ens_pred <- rowMeans(nets)
# df$ens_var <- rowMeans((nets - df$ens_pred)**2)
# mdl <- fit_gamm(df)

