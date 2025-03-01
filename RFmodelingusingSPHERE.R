# I. INPUT DATA

library(readxl)
dataSPHERE <- read_xlsx("SPHERE dataset.xlsx")

demographic <- dataSPHERE[,2:21]
demographic$FINTEST2 <- as.numeric(demographic$FINTEST2)

FCI <- dataSPHERE[,22:51]
FMCE <- dataSPHERE[,52:98]
RRMCS <- dataSPHERE[,100:129]
FMCI <- dataSPHERE[,162:191]
MWCS <- dataSPHERE[,192:213]
TCE <- dataSPHERE[,214:239]
STPFASL <- dataSPHERE[,240:272]
SAAR <- dataSPHERE[,273:288]
CLASS <- dataSPHERE[,290:331]

FCIkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "FCI")
FMCEkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "FMCE")
RRMCSkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "RRMCS")
FMCIkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "FMCI")
MWCSkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "MWCS")
TCEkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "TCE")
STPFASLkey <- read_xlsx("SPHERE answer keys.xlsx", sheet = "STPFASL")

# II. SCORING

library(CTT)
FCI_scored <- score(FCI,FCIkey, output.scored=TRUE)
FMCE_scored <- score(FMCE,FMCEkey, output.scored=TRUE)
RRMCS_scored <- score(RRMCS,RRMCSkey, output.scored=TRUE)
FMCI_scored <- score(FMCI,FMCIkey, output.scored=TRUE)
MWCS_scored <- score(MWCS,MWCSkey, output.scored=TRUE)
TCE_scored <- score(TCE,TCEkey, output.scored=TRUE)
STPFASL_scored <- score(STPFASL,STPFASLkey, output.scored=TRUE)

# III. MACHINE LEARNING IMPLEMENTATION USING RANDOM FOREST

## a. Prepare the SPHERE data

df <- cbind(demographic[,-1],FCI_scored$score,FMCE_scored$score,RRMCS_scored$score,
            FMCI_scored$score,MWCS_scored$score,TCE_scored$score,STPFASL_scored$score,
            apply(SAAR,1,sum),apply(CLASS,1,sum))

colnames(df)[20:28] <- c("FCI","FMCE","RRMCS","FMCI","MWCS","TCE","STPFASL","SAAR","CLASS")

## b. Labeling students' performance in the end of the second semester

cutscore <- 70 # this value was determined by the aggrement of physics teachers participated in this study. Actually, it should be set to increase the prediction performance.
df$Target <- ifelse(df$FINTEST2>cutscore,1,0)

## c. Define data types
df$SCH <- as.factor(df$SCH)
df$COH <- as.factor(df$COH)
df$GDR <- as.factor(df$GDR)
df$FATHOCC <- as.factor(df$FATHOCC)
df$MOTHOCC <- as.factor(df$MOTHOCC)
df$FATHEDU <- as.factor(df$FATHEDU)
df$MOTHEDU <- as.factor(df$MOTHEDU)
df$FATHINC <- as.factor(df$FATHINC)
df$MOTHINC <- as.factor(df$MOTHINC)
df$FATHOCC <- as.factor(df$FATHOCC)
df$SIBL <- as.factor(df$SIBL)
df$DOM <- as.factor(df$DOM)
df$LIT1 <- as.factor(df$LIT1)
df$LIT2 <- as.factor(df$LIT2)
df$PHYIDE1 <- as.factor(df$PHYIDE1)
df$PHYIDE2 <- as.factor(df$PHYIDE2)
df$Target <- as.factor(df$Target)

## d. MODEL RF 1 (ALL)
### 1. Training & Testing
library(randomForest)
library(caret)
library(caTools)
library(pROC)

set.seed(86)

folds <- createFolds(df$Target, k = 10)
metric_list <- sapply(folds, function(fold) {
  df.Train <- df[-fold, ]
  df.Test <- df[fold, ]
  
  rf_1 <- randomForest(Target ~ GDR + AGE + FATHOCC + MOTHOCC + FATHEDU + MOTHEDU + FATHINC + MOTHINC + SIBL + DOM
                       + LIT1 + LIT2 + PHYIDE1 + PHYIDE2 + FCI + FMCE + RRMCS + FMCI + MWCS + TCE + STPFASL + SAAR + CLASS, 
                       data = df.Train, ntree = 500, importance = T)
  Pred.rf_1 <- predict(rf_1, df.Test)
  cf_table <- confusionMatrix(Pred.rf_1, df.Test$Target)
  
  Pred.new.rf_1<-predict(rf_1, newdata = df.Test, type = 'prob')
  roc.mod.1<-roc(df.Test$Target, Pred.new.rf_1[,2], ci = T)
  
  metric <- data.frame("AUROCL" = roc.mod.1$ci[1],
                       "AUROC" = roc.mod.1$auc,
                       "AUROCU" = roc.mod.1$ci[3],
                       "AccuracyL" = cf_table$overall['AccuracyLower'],
                       "Accuracy" = cf_table$overall['Accuracy'],
                       "AccuracyU" = cf_table$overall['AccuracyUpper'],
                       "Sensitivity" = cf_table$byClass['Sensitivity'],
                       "Specificity" = cf_table$byClass['Specificity'])
  return(as.numeric(metric))
})

data.frame("Metrics" = c("AUROCLower", "AUROC", "AUROCUpper", "AccuracyLower", "Accuracy", "AccuracyUpper", "Sensitivity", "Specificity"), "Mean" = rowMeans (metric_list), metric_list)

## e. MODEL RF 2 (RBAs)

### 1. Training & Testing
folds <- createFolds(df$Target, k = 10)
metric_list <- sapply(folds, function(fold) {
  df.Train <- df[-fold, ]
  df.Test <- df[fold, ]
  
  rf_2 <- randomForest(Target ~ FCI + FMCE + RRMCS + FMCI + MWCS + TCE + STPFASL + SAAR + CLASS, 
                       data = df.Train, ntree = 500, importance = T)
  Pred.rf_2 <- predict(rf_2, df.Test)
  cf_table <- confusionMatrix(Pred.rf_2, df.Test$Target)
  
  Pred.new.rf_2<-predict(rf_2, newdata = df.Test, type = 'prob')
  roc.mod.2<-roc(df.Test$Target, Pred.new.rf_2[,2], ci = T)
  
  metric <- data.frame("AUROCL" = roc.mod.2$ci[1],
                       "AUROC" = roc.mod.2$auc,
                       "AUROCU" = roc.mod.2$ci[3],
                       "AccuracyL" = cf_table$overall['AccuracyLower'],
                       "Accuracy" = cf_table$overall['Accuracy'],
                       "AccuracyU" = cf_table$overall['AccuracyUpper'],
                       "Sensitivity" = cf_table$byClass['Sensitivity'],
                       "Specificity" = cf_table$byClass['Specificity'])
  return(as.numeric(metric))
})

data.frame("Metrics" = c("AUROCLower", "AUROC", "AUROCUpper", "AccuracyLower", "Accuracy", "AccuracyUpper", "Sensitivity", "Specificity"), "Mean" = rowMeans (metric_list), metric_list)

## f. MODEL RF 3 (DEMOGRAPHIC)

### 1. Training & Testing
folds <- createFolds(df$Target, k = 10)
metric_list <- sapply(folds, function(fold) {
  df.Train <- df[-fold, ]
  df.Test <- df[fold, ]
  
  rf_3 <- randomForest(Target ~ GDR + AGE + FATHOCC + MOTHOCC + FATHEDU + MOTHEDU + FATHINC + MOTHINC + SIBL + DOM +
                         LIT1 + LIT2 + PHYIDE1 + PHYIDE2, data = df.Train, ntree = 500, importance = T)
  Pred.rf_3 <- predict(rf_3, df.Test)
  cf_table <- confusionMatrix(Pred.rf_3, df.Test$Target)
  
  Pred.new.rf_3<-predict(rf_3, newdata = df.Test, type = 'prob')
  roc.mod.3<-roc(df.Test$Target, Pred.new.rf_3[,2], ci = T)
  
  metric <- data.frame("AUROCL" = roc.mod.3$ci[1],
                       "AUROC" = roc.mod.3$auc,
                       "AUROCU" = roc.mod.3$ci[3],
                       "AccuracyL" = cf_table$overall['AccuracyLower'],
                       "Accuracy" = cf_table$overall['Accuracy'],
                       "AccuracyU" = cf_table$overall['AccuracyUpper'],
                       "Sensitivity" = cf_table$byClass['Sensitivity'],
                       "Specificity" = cf_table$byClass['Specificity'])
  return(as.numeric(metric))
})

data.frame("Metrics" = c("AUROCLower", "AUROC", "AUROCUpper", "AccuracyLower", "Accuracy", "AccuracyUpper", "Sensitivity", "Specificity"), "Mean" = rowMeans (metric_list), metric_list)

## g. MODEL RF 4 (COMBINED)
### 1. Training & Testing
folds <- createFolds(df$Target, k = 10)
metric_list <- sapply(folds, function(fold) {
  df.Train <- df[-fold, ]
  df.Test <- df[fold, ]
  
  rf_4 <- randomForest(Target ~ FMCI + MWCS + FMCE + STPFASL + CLASS + 
                         FATHINC + FATHOCC + MOTHINC + MOTHOCC + FATHEDU, 
                       data = df.Train, ntree = 500, importance = T)
  Pred.rf_4 <- predict(rf_4, df.Test)
  cf_table <- confusionMatrix(Pred.rf_4, df.Test$Target)
  
  Pred.new.rf_4<-predict(rf_4, newdata = df.Test, type = 'prob')
  roc.mod.4<-roc(df.Test$Target, Pred.new.rf_4[,2], ci = T)
  
  metric <- data.frame("AUROCL" = roc.mod.4$ci[1],
                       "AUROC" = roc.mod.4$auc,
                       "AUROCU" = roc.mod.4$ci[3],
                       "AccuracyL" = cf_table$overall['AccuracyLower'],
                       "Accuracy" = cf_table$overall['Accuracy'],
                       "AccuracyU" = cf_table$overall['AccuracyUpper'],
                       "Sensitivity" = cf_table$byClass['Sensitivity'],
                       "Specificity" = cf_table$byClass['Specificity'])
  return(as.numeric(metric))
})

data.frame("Metrics" = c("AUROCLower", "AUROC", "AUROCUpper", "AccuracyLower", "Accuracy", "AccuracyUpper", "Sensitivity", "Specificity"), "Mean" = rowMeans (metric_list), metric_list)

## h. Teacher prediction performance
### 1. Prediction performance of physics teachers
confusionMatrix(as.factor(df$TEACHPRED), df$Target)

### 2. ROC analysis
roc.mod.5<-roc(df$Target, df$TEACHPRED, ci = T)
plot.roc(roc.mod.5,print.thres = F, print.auc = T, legacy.axes = T)

## i. Variable importance analysis of RF2 and RF3
rf_2 <- randomForest(Target ~ FCI + FMCE + RRMCS + FMCI + MWCS + TCE + STPFASL + SAAR + CLASS, 
                     data = df.Train, ntree = 500, importance = T)
rf_3 <- randomForest(Target ~ SCH + COH + GDR + AGE + FATHOCC + MOTHOCC + FATHEDU + MOTHEDU + FATHINC + MOTHINC + SIBL + DOM +
                       LIT1 + LIT2 + PHYIDE1 + PHYIDE2, 
                     data = df.Train, ntree = 500, importance = T)

par(mfrow = c(1, 2))
varImpPlot(rf_2, main = "RF 2", type = 2)
varImpPlot(rf_3, main = "RF 3", type = 2)