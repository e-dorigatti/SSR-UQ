library(dplyr)
library(mgcv)

df <- read.csv2("data/ISIC_2020_Training_GroundTruth_v2.csv", sep=",")

### remove empty
keep_mask <- (df$sex != "") & (df$anatom_site_general_challenge != "") & !is.na(df$age_approx)
cat("removing", sum(!keep_mask), "rows with empty columns\n")
df <- df[keep_mask,]

### encode
df$sex <- factor(df$sex)
df$diagnosis <- factor(df$diagnosis)
df$site <- factor(df$anatom_site_general_challenge)
df$benign_malignant <- factor(df$benign_malignant)
df$patient_id <- factor(df$patient_id)

pats <- df %>% group_by(patient_id) %>% summarise(n=n()) %>% filter(n>=4)
df <- df[df$patient_id %in% pats$patient_id,]
cat("kept", nrow(df), "lesions from patients with at least four\n")
pats <- pats[sample(1:nrow(pats)),]

test_pats <- pats[1:170, "patient_id"]
tune_pats <- pats[171:340, "patient_id"]
train_pats <- pats[341:nrow(pats), "patient_id"]

test_df <- df %>% filter(patient_id %in% test_pats$patient_id)
tune_df <- df %>% filter(patient_id %in% tune_pats$patient_id)
train_df <- df %>% filter(patient_id %in% train_pats$patient_id)

cat("got", nrow(test_df), "patients for test\n")
cat("got", nrow(tune_df), "patients for tuning\n")
cat("got", nrow(train_df), "patients for training\n")

test_df$subset <- "test"
tune_df$subset <- "tune"
train_df$subset <- "trainval"

df_all <- rbind(train_df, tune_df, test_df)
saveRDS(df_all, "data/train-processed.RDS")

# model matrix for structured effects
#mdl <- gam(target ~ site + sex + s(age_approx), family = "binomial", data = df)
mdl <- bam(
    target ~ site + sex + s(age_approx, by=sex),
    family = "binomial", data = df_all, discrete = TRUE, nthreads = 4
)

x_struc <- as.data.frame(model.matrix(mdl))
x_struc$target <- df_all$target
x_struc$image <- df_all$image
x_struc$patient_id <- df_all$patient_id
x_struc$subset <- df_all$subset

write.csv2(x_struc, "data/x_struc.csv")

