# The Students' Performance in Physics Education Research (SPHERE) Dataset

This repository hosts tutorials for random forest modeling using SPHERE dataset in R. This project is aimed to analyze students' performance in physics as reported by the SPHERE dataset. The prediction case is whether students will pass/ fail at the end of the semester. Random forest will be trained to do this prediction task. The prediction performance will be evaluated using some metrics including area under receiver operating characteristic (ROC) curve (AUROC), accuracy, sensitivity, and specificity. The code was written in R version 4.4.1. Ensure you have downloaded compatible R version to run this script.

## Steps for training the random forest model using SPHERE dataset
**1. Download the SPHERE dataset**

Use the DOI link from the Mendeley Data repository to access the SPHERE dataset: (https://doi.org/10.17632/88d7m2fv7p.1). The raw data will be tabular in XLSX or CSV format.

**2. Answer keys to score the students' performance**

This study employed research based assessments (RBAs) to establish the SPHERE dataset. We need answer keys to score students' performance on each test administered in this study. Unfortunately, it can be published here. If you are physics educators, the PhysPort platform is the home of RBAs (https://www.physport.org). You should sign in (or sign up if you don't have an account) to the platform as educators. This account type will provide a full access to the items examined by the RBAs including the answer keys.

**3. Install the dependencies**

This code was written under some packages such as `readxl`, `randomForest`, `caret`, `caTools`, and `pROC`. You should install those packages before running the code.

**3. Run the Scripts**

Open the project, **SPHERE random forest.Rproj**, in RStudio. Follow the instructions in the script to prepare your data set. Run the program.

## Related Publications
Santoso, Purwoko Haryadi; Setiaji, Bayu; Kurniawan, Yohanes; Yudi, Wahyudi; Bahri, Syamsul; Fathurrahman, Fathurrahman; Kusuma, Mobinta; Wusqo, Indah Urwatin; Muldayanti, Nuri Dewi; Kurniawan, Arif Didik; Syahbrudin, Johan (2024), “SPHERE: Students' performance dataset of conceptual understanding, scientific ability, and learning attitude in physics education research (PER)”, Mendeley Data, V1, https://doi.org/10.17632/88d7m2fv7p.1

