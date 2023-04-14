library(dplyr)
library(pROC)
library(reshape2)
library(mgcv)
library(gridExtra)
library(Matrix)


print_mod_summary <- function(mm) {
  ss <- summary(mm)
  aa <- anova(mm)
  pvs <- aa$pTerms.pv
  names(pvs) <- NULL
  cat("pen,aRsq,dev,freml,pvals(s),pvals(u)", pen, ss$r.sq, ss$dev.expl, ss$sp.criterion, pvs, aa$s.pv, "\n")
}


# with fixed ensemble
#pen <- 0.0002
all_preds <- read.csv2("ensemble_preds_heldout_nost.csv", sep = ",", dec = ".")
for(pen in unique(all_preds$penalty)) {
  preds <- all_preds[all_preds$penalty == pen,]
  df <- readRDS("data/train-processed.RDS")[preds$idx + 1,]
  preds$outcome <- df$target
  nets <- preds[,grep("network_", colnames(preds))]
  nets <- nets[,colSums(is.na(nets)) < nrow(nets)]
  df$ens_pred <- rowMeans(nets)

  print_mod_summary(
    mdl<-bam(target ~ site + sex + s(age_approx,by=sex) + s(patient_id, bs = "re") + offset(ens_pred),
            family = "binomial", data = df[df$subset == "tune",], discrete = TRUE, nthreads = 6)
  )
  print_mod_summary(
   bam(target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re") + offset(ens_pred),
       family = "binomial", data = df[df$subset == "test",], discrete = TRUE, nthreads = 6)
  )
}

# without dnn at all
print_mod_summary(
  bam(target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re"),
      family = "binomial", data = df[df$subset == "tune",], discrete = TRUE, nthreads = 6)
)
print_mod_summary(
  bam(target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re"),
      family = "binomial", data = df[df$subset == "test",], discrete = TRUE, nthreads = 6)
)


#plot(readRDS(paste0("mixmod_test_pen=", "0.1", ".RDS")))



# with dnn uncertainty
for(pen in unique(all_preds$penalty)) {
  print_mod_summary(readRDS(paste0("mixmod_tune_pen=", pen, ".RDS")))
  #print_mod_summary(readRDS(paste0("mixmod_test_pen=", pen, ".RDS")))
}

# with fixed single networks, no uncertainty, best penalty
all_preds <- read.csv2("ensemble_preds_heldout.csv", sep = ",", dec = ".")
preds <- all_preds[all_preds$penalty == 0.0002,]
df <- readRDS("data/train-processed.RDS")[preds$idx + 1,]
df$patient_id <- as.factor(df$patient_id)
preds$outcome <- df$target
net_cols <- colnames(preds)[grep("network_", colnames(preds))]
for(c in net_cols) {
  df$net_pred <- preds[,c]
  print_mod_summary(
    bam(target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re") + offset(net_pred),
        family = "binomial", data = df[df$subset == "tune",], discrete = TRUE, nthreads = 6))
  print_mod_summary(
    bam(target ~ site + sex + s(age_approx, by=sex) + s(patient_id, bs = "re") + offset(net_pred),
        family = "binomial", data = df[df$subset == "test",], discrete = TRUE, nthreads = 6))
}


# with ensemble and uncertainty -------------------------------------------


mdl_mix_unc_ens = list(
  "1e-04" = readRDS("mixmod_4lesions_pen=1e-04.RDS"),
  "2e-04" = readRDS("mixmod_4lesions_pen=2e-04.RDS"),
  "5e-04" = readRDS("mixmod_4lesions_pen=5e-04.RDS"),
  "0.001" = readRDS("mixmod_4lesions_pen=0.001.RDS"),
  "0.002" = readRDS("mixmod_4lesions_pen=0.002.RDS"),
  "0.005" = readRDS("mixmod_4lesions_pen=0.005.RDS"),
  "0.01" = readRDS("mixmod_4lesions_pen=0.01.RDS"),
  "0.02" = readRDS("mixmod_4lesions_pen=0.02.RDS"),
  "0.05" = readRDS("mixmod_4lesions_pen=0.05.RDS"),
  "1" = readRDS("mixmod_4lesions_pen=0.1.RDS"),
  "2" = readRDS("mixmod_4lesions_pen=0.2.RDS"),
  "5" = readRDS("mixmod_4lesions_pen=0.5.RDS")
)

for(n in names(mdl_mix_unc_ens)) {
  mm <- mdl_mix_unc_ens[[n]]
  ss <- summary(mm)
  cat("pen,aRsq,dev,freml,", n, ss$r.sq, ss$dev.expl, ss$sp.criterion, "\n")
}

mdl_mix_unc_ens[[]]

# significance of terms
for(n in names(mdl_mix_unc_ens)) {
  mm <- mdl_mix_unc_ens[[n]]
  aa <- anova(mm)
  pvs <- aa$pTerms.pv
  names(pvs) <- NULL
  cat(n, pvs, "\n")
}


