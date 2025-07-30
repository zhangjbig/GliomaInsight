# Welcome to extracmator!
### This model is to identify prognostic radiomics features using clinical information and radiomics features extracted from T2-weighted MRI imagings.

###  The required R packages

library(ggpubr)

library(reshape2)

library("survival")

library("survminer")

library(dplyr)

library(purrr)

library(optparse)

library(data.table)


### The main steps of the pipeline:

#### Train RadioML
Usage: .\RadioML_main.R [options]
Options:
        -f FILE, --file=FILE
                Choose the features data file

        -p PREFIX, --prefix=PREFIX
                Make the prefix of the result, default is prefix

        -t PARTIAL, --partial=PARTIAL
                The ratio of split the samples into train and test (integer: 1-9, default 5)

        -n NUMPERM, --numperm=NUMPERM
                The number of random permutations  (integer, default 100)
##### Examples
Rscript .\RadioML_main.R -f demo_feature_data.txt -p prefix -t 5 -n 1000

#### Extract robust radiomics features and the corresponding coefficients
Usage: .\RadioML_features.R [options]
Options:
        -p PREFIX, --prefix=PREFIX
                make the prefix of the result, default is prefix

        -t PARTIAL, --partial=PARTIAL
                The ratio of split the samples into train and test (integer: 1-9, default 5)

        -n NUMPERM, --numperm=NUMPERM
                The number of random permutations  (integer, default 100, it should be the same with RadioML_main.R option -n)

        -R RDATA, --RData=RDATA
                the previous main outout RData
##### Examples
Rscript .\RadioML_features.R -p TIANTAN -t 5 -n 1000 -R TIANTAN_1000_5_training.RData
Rscript .\RadioML_features.R -p TIANTAN -t 5 -n 1000 -R prefix_1000_5_training.RData


library(ggpubr)
library(reshape2)
library("survival")
library("survminer")
library(dplyr)
library(purrr)
library(optparse)
library(data.table)


option_list <- list(
  make_option(c("-f", "--file"), type = "character", default=FALSE,
              help="Choose the feature and samples dataframe"),
  make_option(c("-p", "--prefix"), type="character", default=FALSE,
              help="make the prefix of the result, default is prefix"),
  make_option(c("-c", "--features"), type="character", default=FALSE,
              help="Choose the feature and coe data"),
  make_option(c("-m", "--methods"), type="character", default=FALSE,
              help="marker the previous methods")	  
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

prefix <- "TIANTAN"
files <- "demo_feature_data.txt"
feature <- "perm5.20.feature_coe.txt"

files <- "demo.txt"#opt$file
prefix <- opt$prefix
feature <- "perm5.20.feature_coe.txt"#opt$features
methods <- opt$methods


feature_coe_final <- read.table(feature, head = T, row.names = 1)
feature_final <- rownames(feature_coe_final) #coe中的行索引给feature_final
feature_final <- gsub("\\.", "-", feature_final)
feature_coe_final <- feature_coe_final[,1]
names(feature_coe_final) <- feature_final

# TIANTAN score
datasets <- "TIANTAN"
feature_data <- read.table(files)

#print(feature_data) #第一列正常

feature_data1 <- apply(feature_data, 2, as.numeric) #产生NA
print(feature_data1)

rownames(feature_data1) <- rownames(feature_data[0])
feature_data <- feature_data1

#print(rownames(feature_data)) #打印出来是数字

setdiff(feature_final, row.names(feature_data) )

feature_data1 <- apply(feature_data, 2, as.numeric) #强制改变过程中产生了NA
#print(feature_data1)


setdiff(feature_final, row.names(feature_data) )

score_final <- NULL
for(i in 1:dim(feature_data)[2]) {
  score_final_tmp <- 0
  for(j in 1:length(feature_final)){
    score_final_tmp <- score_final_tmp + feature_data[which( rownames(feature_data) == feature_final[j]),i] * feature_coe_final[feature_final[j]]
  }
  score_final_tmp <- data.matrix(score_final_tmp)
  rownames(score_final_tmp) <- colnames(feature_data)[i]
  colnames(score_final_tmp) <- "score"
  score_final <- rbind(score_final, score_final_tmp)
  print(length(score_final_tmp)) #1
  print(ncol(feature_data)) #167
  
}
#Error in dimnames(x) <- dn : 'dimnames'的长度[1]必需与陈列范围相等
print(length(score_final_tmp)) #1
print(ncol(feature_data)) #167
# RF OS cutoff

score_final_CGGA <- data.frame(score_final)
cutoff_RF_OS <- median(score_final_CGGA$score)

score_final_CGGA$group <- 1
score_final_CGGA$group[which(score_final_CGGA$score > cutoff_RF_OS)] <- 2
#score_final_CGGA$group[which(score_final_CGGA$score > median(score_final_CGGA$score))] <- 2
#score_final_CGGA$Censor <- t(CGGA["Censor",rownames(score_final_CGGA)])
#score_final_CGGA$OS <- t(CGGA["OS",rownames(score_final_CGGA)])

# not use the previous clinical data, use new tmp <- as.data.frame(t(CGGA))

tmp <- t(feature_data[1:2,])
tmp <- data.frame(tmp)
tmp$OS <- as.numeric(tmp$OS)
tmp$Censor <- tmp$OS_Censor
tmp <- tmp[,-2]
score_final_CGGA <- merge(score_final_CGGA, tmp, by = "row.names")
rownames(score_final_CGGA) <- score_final_CGGA[,1]
score_final_CGGA <- score_final_CGGA[,-1]
fit <- survfit(Surv( OS, Censor ) ~ group, data = score_final_CGGA)
surv_diff <- survdiff(Surv(OS, Censor) ~ group, data = score_final_CGGA)
p.KM <- 1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1)

p <- ggsurvplot(fit,
                conf.int = FALSE,
                pval = FALSE,
                legend.title =
                  paste("logrank p = ",signif(p.KM, 3)),
                legend.labs = 
                  c(paste("low score = ",table(score_final_CGGA[["group"]])[1]), paste("high score = ",table(score_final_CGGA[["group"]])[2])),
                risk.table.col = "group", # Change risk table color by groups
                ggtheme = theme_classic(), # Change ggplot2 theme
                palette = c("#E7B800", "#2E9FDF"))

pdf( paste0(datasets, ".", methods, "_OS_KM.pdf"), width = 6, height = 6)
print(p$plot, newpage = FALSE)
dev.off()


feature_score <- list(score_final_CGGA)
names(feature_score) <- c("TIANTAN")

# output table
for(sample in names(feature_score) ) {
  
  feature_tmp <-  t(feature_data)
  feature_tmp <- data.frame(feature_tmp)
  colnames(feature_tmp) <- gsub( "\\.", "-", colnames(feature_tmp))
  tmp_feature_count <- feature_tmp[,colnames(feature_tmp) %in% feature_final]
  
  score_final_tmp <- feature_score[[sample]]
  table <- merge(score_final_tmp, tmp_feature_count, by="row.names" )
  write.table(table, paste0( sample, ".", methods, "_table_feature.txt"), sep = "\t", quote = F, row.names = F)
  
}

# save the score_feature_final
save(feature_score, feature_coe_final, feature_final, file = paste0(methods, ".score.RData") )


#### Calculate the Risk score of the samples
Usage: .\Risk_score.R [options]
Options:
        -f FILE, --file=FILE
                Choose the feature and samples dataframe

        -p PREFIX, --prefix=PREFIX
                Make the prefix of the result, default is prefix

        -c FEATURES, --features=FEATURES
                Choose the feature and coe data

        -m METHODS, --methods=METHODS
                Marke the previous methods
##### Examples
Rscript .\Risk_score.R -f demo_feature_data.txt -p prefix -c perm5.20.feature_coe.txt -m perm5.20
Rscript .\Risk_score.R -f demo.txt -p prefix -c perm5.20.feature_coe.txt -m perm5.20

```ruby
library(ggpubr)
library(reshape2)
library("survival")
library("survminer")
library(dplyr)
library(purrr)
library(optparse)
library(data.table)

option_list <- list(
  make_option(c("-f", "--file"), type = "character", default=FALSE,
              help="Choose the feature and samples dataframe"),
  make_option(c("-p", "--prefix"), type="character", default=FALSE,
              help="make the prefix of the result, default is prefix"),
  make_option(c("-c", "--features"), type="character", default=FALSE,
              help="Choose the feature and coe data"),
  make_option(c("-m", "--methods"), type="character", default=FALSE,
              help="marker the previous methods")	  
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

prefix <- "TIANTAN"
files <- "demo.txt"
feature <- "perm5.20.feature_coe.txt"

files <- opt$file
prefix <- opt$prefix
feature <- opt$features
methods <- opt$methods


feature_coe_final <- read.table(feature, head = T, row.names = 1)
feature_final <- rownames(feature_coe_final)
feature_final <- gsub("\\.", "-", feature_final)
feature_coe_final <- feature_coe_final[,1]
names(feature_coe_final) <- feature_final

# TIANTAN score
datasets <- "TIANTAN"
feature_data <- read.table(files)
feature_data1 <- apply(feature_data, 2, as.numeric)
rownames(feature_data1) <- rownames(feature_data)
feature_data <- feature_data1

setdiff(feature_final, row.names(feature_data) )


score_final <- NULL
for(i in 1:dim(feature_data)[2]) {
  score_final_tmp <- 0
  for(j in 1:length(feature_final)){
    score_final_tmp <- score_final_tmp + feature_data[which( rownames(feature_data) == feature_final[j]),i] * feature_coe_final[feature_final[j]]
  }
  score_final_tmp <- data.matrix(score_final_tmp)
  rownames(score_final_tmp) <- colnames(feature_data)[i]
  colnames(score_final_tmp) <- "score"
  score_final <- rbind(score_final, score_final_tmp)
  print(length(score_final_tmp)) #1
  print(ncol(feature_data)) #167
  
}

print(length(score_final_tmp)) #1
print(ncol(feature_data)) #167
# RF OS cutoff

score_final_CGGA <- data.frame(score_final)
cutoff_RF_OS <- median(score_final_CGGA$score)

score_final_CGGA$group <- 1
score_final_CGGA$group[which(score_final_CGGA$score > cutoff_RF_OS)] <- 2
#score_final_CGGA$group[which(score_final_CGGA$score > median(score_final_CGGA$score))] <- 2
#score_final_CGGA$Censor <- t(CGGA["Censor",rownames(score_final_CGGA)])
#score_final_CGGA$OS <- t(CGGA["OS",rownames(score_final_CGGA)])

# not use the previous clinical data, use new tmp <- as.data.frame(t(CGGA))

tmp <- t(feature_data[1:2,])
tmp <- data.frame(tmp)
tmp$OS <- as.numeric(tmp$OS)
tmp$Censor <- tmp$OS_Censor
tmp <- tmp[,-2]
score_final_CGGA <- merge(score_final_CGGA, tmp, by = "row.names")
rownames(score_final_CGGA) <- score_final_CGGA[,1]
score_final_CGGA <- score_final_CGGA[,-1]
fit <- survfit(Surv( OS, Censor ) ~ group, data = score_final_CGGA)
surv_diff <- survdiff(Surv(OS, Censor) ~ group, data = score_final_CGGA)
p.KM <- 1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1)

p <- ggsurvplot(fit,
                conf.int = FALSE,
                pval = FALSE,
                legend.title =
                  paste("logrank p = ",signif(p.KM, 3)),
                legend.labs = 
                  c(paste("low score = ",table(score_final_CGGA[["group"]])[1]), paste("high score = ",table(score_final_CGGA[["group"]])[2])),
                risk.table.col = "group", # Change risk table color by groups
                ggtheme = theme_classic(), # Change ggplot2 theme
                palette = c("#E7B800", "#2E9FDF"))

pdf( paste0(datasets, ".", methods, "_OS_KM.pdf"), width = 6, height = 6)
print(p$plot, newpage = FALSE)
dev.off()


feature_score <- list(score_final_CGGA)
names(feature_score) <- c("TIANTAN")

# output table
for(sample in names(feature_score) ) {
  
  feature_tmp <-  t(feature_data)
  feature_tmp <- data.frame(feature_tmp)
  colnames(feature_tmp) <- gsub( "\\.", "-", colnames(feature_tmp))
  tmp_feature_count <- feature_tmp[,colnames(feature_tmp) %in% feature_final]
  
  score_final_tmp <- feature_score[[sample]]
  table <- merge(score_final_tmp, tmp_feature_count, by="row.names" )
  write.table(table, paste0( sample, ".", methods, "_table_feature.txt"), sep = "\t", quote = F, row.names = F)
  
}

# save the score_feature_final
save(feature_score, feature_coe_final, feature_final, file = paste0(methods, ".score.RData") )
```

```ruby


# -*- coding: utf-8 -*-
if(FALSE) {
"
Created on Fri Oct 25 00:23:49 2020

@author: LL,JZ, email: jz2716@buaa.edu.cn
"
}
library(optparse)
library(data.table)
library("survival")
library("survminer")
library(ggpubr)
library(reshape2)
library(dplyr)
library(purrr)

option_list <- list(
  make_option(c("-f", "--file"), type = "character", default=FALSE,
              help="Choose the features data file"),
  make_option(c("-p", "--prefix"), type="character", default=FALSE,
              help="make the prefix of the result, default is prefix"),
  make_option(c("-t", "--partial"), type = "integer", default=FALSE,
              help="the ratio of split the samples into train and test (integer: 1-9, default 5)"),
  make_option(c("-n", "--numperm"), type="integer", default=FALSE,
              help="the repeat times  (integer, default 100)")
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);


#files <- "demo_feature_data.txt"
files <- opt$files
prefix <- opt$prefix
partial1 <- opt$partial
num_perm <- opt$numperm


if(!num_perm) num_perm <- 100
if(!partial1) partial1 <- 5
if(prefix == FALSE ) prefix <- "prefix"
if(files == FALSE ) {
print("file is empty, it is need the train datasets, use the help to find the details")
cmd = paste("Rscript RadioML_main.R", "-f", opt$file, "-p",  opt$prefix, "-t", opt$partial, "-n", opt$numperm)

cat(cmd, "\n")
stop()

}
#argv <- commandArgs(TRUE)
#prefix <- argv[1]
#partial1 <- argv[2]
#partial1 <- as.numeric(partial1)
#files <- argv[3]
#prefix <- "test"



print(prefix)
print(partial1)
print(files)
print(num_perm)


training <- read.table(files, head = T, row.names = 1)
training["OS_Censor",] <- training["OS_Censor",] + 1
training<-data.matrix(training)
samples <- length(colnames(training))
partial <- round(samples/10*partial1)
clinical_number <- 3
matrix <- matrix(0,nrow = dim(training)-clinical_number, ncol = num_perm)
table_p <- data.frame(matrix)
rownames(table_p) <- rownames(training)[(clinical_number + 1):dim(training)[1]]
table_HR <- data.frame(matrix)
rownames(table_HR) <- rownames(training)[(clinical_number + 1):dim(training)[1]]
table_coe <- data.frame(matrix)
rownames(table_coe) <- rownames(training)[(clinical_number + 1):dim(training)[1]]

table_test <- matrix(0,nrow = 1, ncol = num_perm)
rownames(table_test) <- c("test_result")

list_perm <- list()
result <- list()
sink(paste0(prefix, "_loop_info.txt"))
set.seed(10001)
for(i in 1:num_perm ) {
  print(i)
  tmp <- ""
  tmp <- sample(samples)
  train <- ""
  train <- t(training[,tmp[1:partial]])
  test <- t(training[,tmp[(partial+1):samples]])
  list_perm[[i]] <- list(train = train, test  = test)
  
  for(j in (clinical_number + 1):dim(training)[1]) {
    mdf <- paste0("i ", i," : j ",j)
    print(mdf)
    #train[,2:length(colnames(train))] <- apply(train[,2:length(colnames(train))], 2, as.numeric) 
    res <- coxph(Surv(OS, OS_Censor) ~ train[,j], data = data.frame(train))
    #result[[j]] <- tryCatch(res <- coxph(Surv(OS, OS_Censor) ~ train[,j], data = data.frame(train)),  error = function(e) paste("something wrong here"))
    summcph <- summary(res)
    table_p[j-clinical_number,i] <- summcph$coefficients[5]
    table_HR[j-clinical_number,i] <- summcph$coefficients[2]
    table_coe[j-clinical_number,i] <- summcph$coefficients[1]
    
  }
  features <-""
  features <- intersect(rownames(table_p)[which(table_p[,i] <= 0.05)],rownames(table_p)[which(table_HR[,i] > 1)])
  
  if(length(features) == 0) {
  next
  }
  
  test_feature <- test[,features]
  score_test<-NULL
  for( j in 1: dim(test_feature)[1]){
    score_j<-0
    for(k in 1:dim(test_feature)[2]) {
      score_j<-score_j+test_feature[j,k]*table_coe[colnames(test_feature)[k],i]
    }
    score_j<-data.matrix(score_j)
    rownames(score_j)<-rownames(test_feature)[j]
    colnames(score_j)<-"score"
    score_test<-rbind(score_test, score_j)
  }
  
  score_test <- data.frame(score_test)
  score_test$group <- 1 
  score_test$group[which(score_test$score > median(score_test$score))] <- 2
  #score_test$OS_Censor <- t(training["OS_Censor",rownames(score_test)])
  #score_test$OS <- t(training["OS",rownames(score_test)])
  score_test$OS_Censor <- training["OS_Censor",rownames(score_test)]
  score_test$OS <- training["OS",rownames(score_test)]
  
  fit <- survfit(Surv(OS, OS_Censor) ~ group, data = score_test)
  surv_diff <- survdiff(Surv(OS, OS_Censor) ~ group, data = score_test)
  p.KM <- round(1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1),3)
  
  #p <- ggsurvplot(fit,
  #                conf.int = FALSE,
  #               pval = FALSE,
  #               legend.title =
  #                  paste("logrank p = ",round(p.KM,3)),
  #                legend.labs = 
  #                  c(paste("low score = ",table(score_test[["group"]])[1]), paste("high score = ",table(score_test[["group"]])[2])),
  #                risk.table.col = "group", # Change risk table color by groups
  #                ggtheme = theme_classic(), # Change ggplot2 theme
  #                palette = c("#E7B800", "#2E9FDF"))
  
  table_test[1,i] <- 1 - pchisq(surv_diff$chisq, length(surv_diff$n) - 1)
  print(i)
}
sink()

table_coe_test <- matrix(0,nrow=dim(table_coe)[1], ncol=dim(table_coe)[2])
rownames(table_coe_test) <- rownames(table_coe)
colnames(table_coe_test) <- colnames(table_coe)

for( i in 1: dim(table_test)[2]) {
  if (table_test[1,i] <= 0.05) {
    features_tmp<-""
    features_tmp <-intersect(rownames(table_p)[which(table_p[,i] <= 0.05)],rownames(table_p)[which(table_HR[,i] > 1)])
    table_coe_test[as.character(features_tmp),i] <- 1
  }
}
assign(paste0(prefix, "_table_coe_test"), table_coe_test)
assign(paste0(prefix, "_table_coe"), table_coe)
write.table(table_coe_test, paste0(prefix, "_table_coe_test.txt"), sep = "\t", quote = F)
write.table(table_coe, paste0(prefix, "_table_coe.txt"), sep = "\t", quote = F)
save.image(paste0(prefix, "_", num_perm, "_", partial1, "_training.RData"))


#加载R对象
load(file="prefix_100_5_training.RData")

#加载某个脚本中获得的 所有R对象
load(file="prefix_100_5_training.RData")

dev.off()
```




