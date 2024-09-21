# INPUT DATA

library(readxl)
dataSPHERE <- read_xlsx("SPHERE dataset.xlsx")

demographic <- dataSPHERE[,1:21]
FCI <- dataSPHERE[,22:51]
FMCE <- dataSPHERE[,52:98]
RRMCS <- dataSPHERE[,99:128]
FMCI <- dataSPHERE[,129:158]
MWCS <- dataSPHERE[,159:180]
TCE <- dataSPHERE[,181:206]
STPFASL <- dataSPHERE[,207:239]
SAAR <- dataSPHERE[,240:255]
CLASS <- dataSPHERE[,256:291]

FCIkey <- read_xlsx("answer keys.xlsx", sheet = "FCI")
FMCEkey <- read_xlsx("answer keys.xlsx", sheet = "FMCE")
RRMCSkey <- read_xlsx("answer keys.xlsx", sheet = "RRMCS")
FMCIkey <- read_xlsx("answer keys.xlsx", sheet = "FMCI")
MWCSkey <- read_xlsx("answer keys.xlsx", sheet = "MWCS")
TCEkey <- read_xlsx("answer keys.xlsx", sheet = "TCE")
STPFASLkey <- read_xlsx("answer keys.xlsx", sheet = "STPFaSL")

# SCORING

library(CTT)
FCI_scored <- score(FCI,FCIkey, output.scored=TRUE)
FMCE_scored <- score(FMCE,FMCEkey, output.scored=TRUE)
RRMCS_scored <- score(RRMCS,RRMCSkey, output.scored=TRUE)
FMCI_scored <- score(FMCI,FMCIkey, output.scored=TRUE)
MWCS_scored <- score(MWCS,MWCSkey, output.scored=TRUE)
TCE_scored <- score(TCE,TCEkey, output.scored=TRUE)
STPFASL_scored <- score(STPFASL,STPFASLkey, output.scored=TRUE)

# MACHINE LEARNING IMPLEMENTATION USING RANDOM FOREST

## Prepare the SPHERE data

df <- cbind(demographic,FCI_scored$score,FMCE_scored$score,RRMCS_scored$score,FMCI_scored$score,
            MWCS_scored$score,TCE_scored$score,STPFASL_scored$score,apply(SAAR,1,sum),apply(CLASS,1,sum))

colnames(df)[22:30] <- c("FCI","FMCE","RRMCS","FMCI","MWCS","TCE","STPFASL","SAAR","CLASS")

## Labeling students' performance in the end of the second semester

cutscore <- 70
df$Target <- ifelse(df$FINTEST2>cutscore,1,0)

## Define data types

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
df$FINTEST2 <- as.numeric(df$FINTEST2)
df$Target <- as.factor(df$Target)

## Data splitting

library(randomForest)
library(caret)
library(caTools)
library(pROC)

set.seed(86)

Ratio <-  0.75
split <- sample.split(df$Target, SplitRatio = Ratio)

df.Train <- subset(df, split == 1)
df.Test <- subset(df, split == 0)

## MODEL RF 1 (ALL)

### Training

rf_1 <- randomForest(Target ~ SCH + COH + GDR + AGE + FATHOCC + MOTHOCC + FATHEDU + MOTHEDU + FATHINC + MOTHINC + SIBL + DOM
                     + LIT1 + LIT2 + PHYIDE1 + PHYIDE2 + FCI + FMCE + RRMCS + FMCI + MWCS + TCE + STPFASL + SAAR + CLASS, 
                     data = df.Train, ntree = 500, importance = T)

### Testing

Pred.rf_1 <- predict(rf_1, df.Test)
confusionMatrix(Pred.rf_1, df.Test$Target)

### ROC Analysis

Pred.new.rf_1<-predict(rf_1, newdata = df.Test, type = 'prob')
roc.mod.1<-roc(df.Test$Target, Pred.new.rf_1[,2], ci = T)
plot.roc(roc.mod.1,print.thres = F, print.auc = T, legacy.axes = T)

## MODEL RF 2 (RBAs)

### Training

rf_2 <- randomForest(Target ~ FCI + FMCE + RRMCS + FMCI + MWCS + TCE + STPFASL + SAAR + CLASS, 
                     data = df.Train, ntree = 500, importance = T)
rf_2

### Testing

Pred.rf_2 <- predict(rf_2, df.Test)
confusionMatrix(Pred.rf_2, df.Test$Target)

### ROC Analysis

Pred.new.rf_2<-predict(rf_2, newdata = df.Test, type = 'prob')
roc.mod.2<-roc(df.Test$Target, Pred.new.rf_2[,2], ci = T)
plot.roc(roc.mod.2,print.thres = F, print.auc = T, legacy.axes = T)

## MODEL RF 3 (DEMOGRAPHIC)

### Training

rf_3 <- randomForest(Target ~ SCH + COH + GDR + AGE + FATHOCC + MOTHOCC + FATHEDU + MOTHEDU + FATHINC + MOTHINC + SIBL + DOM +
                       LIT1 + LIT2 + PHYIDE1 + PHYIDE2, 
                     data = df.Train, ntree = 500, importance = T)
rf_3

### Testing

Pred.rf_3 <- predict(rf_3, df.Test)
confusionMatrix(Pred.rf_3, df.Test$Target)

### ROC Analysis

Pred.new.rf_3<-predict(rf_3, newdata = df.Test, type = 'prob')
roc.mod.3<-roc(df.Test$Target, Pred.new.rf_3[,2], ci = T)
plot.roc(roc.mod.3,print.thres = F, print.auc = T, legacy.axes = T)

## MODEL RF 4 (COMBINED)
### Training

rf_4 <- randomForest(Target ~ FMCI + MWCS + FMCE + STPFASL + CLASS + SCH + COH + FATHINC + FATHOCC + MOTHINC, 
                     data = df.Train, ntree = 500, importance = T)
rf_4

### Testing

Pred.rf_4 <- predict(rf_4, df.Test)
confusionMatrix(Pred.rf_4, df.Test$Target)

### ROC Analysis

Pred.new.rf_4<-predict(rf_4, newdata = df.Test, type = 'prob')
roc.mod.4<-roc(df.Test$Target, Pred.new.rf_4[,2], ci = T)
plot.roc(roc.mod.4,print.thres = F, print.auc = T, legacy.axes = T)

### Variable Importance Analysis

varImpPlot(rf_4, main = "Variable Importance Model RF 4")

## Teacher prediction performance

confusionMatrix(as.factor(df$TEACHPRED), df$Target)

roc.mod.5<-roc(df$Target, df$TEACHPRED, ci = T)
plot.roc(roc.mod.5,print.thres = F, print.auc = T, legacy.axes = T)

par(mfrow = c(1, 2))
varImpPlot(rf_2, main = "RF 2", type = 2)
varImpPlot(rf_3, main = "RF 3", type = 2)