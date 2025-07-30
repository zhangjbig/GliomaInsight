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

files <- opt$file
prefix <- opt$prefix
feature <- opt$features
methods <- opt$methods

#读原数据集
feature <- "perm5.20.feature_coe.txt"
feature_coe_final <- read.table(feature, head = T, row.names = 1)
feature_final <- rownames(feature_coe_final)
feature_final <- gsub("\\.", "-", feature_final)
feature_coe_final <- feature_coe_final[,1]
names(feature_coe_final) <- feature_final

# TIANTAN score
datasets <- "TIANTAN"

feature_data <- read.table("demo_feature_data.txt")
feature_data1 <- apply(feature_data, 2, as.numeric)
rownames(feature_data1) <- rownames(feature_data)
feature_data <- feature_data1
setdiff(feature_final, row.names(feature_data) )
print(feature_data)
#算分数
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
}
# RF OS cutoff分组
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






#读本数据集
prefix <- "TIANTAN"
files <- "demoo.csv"

feature_data <- read.csv(files)
rownames(feature_data)<-feature_data[,1] #将数据框的第一列作为行名
#feature_data<-feature_data[,-1] #将数据框的第一列删除，只留下剩余的列作为数据

print(dim(feature_data)) #demoo37 2

feature_data1 <- apply(feature_data[-1], 2, as.numeric)#只把数值给data1

print(feature_data1) #看看是不是达到了你的要求

rownames(feature_data1) <- rownames(feature_data)
feature_data <- feature_data1
#print(rownames(feature_data))
#print(feature_data) #这样就对了
#算分数
score_final <- NULL
print(dim(feature_data)[2])
for(i in 1:dim(feature_data)[2]) {
  score_final_tmp <- 0
  for(j in 1:length(feature_final)){
    #print(feature_final)
    print(feature_data[which( rownames(feature_data) == feature_final[j]),i])
    score_final_tmp <- score_final_tmp + feature_data[which( rownames(feature_data) == feature_final[j]),i] * feature_coe_final[feature_final[j]]
  }
  #print(score_final_tmp)
  score_final_tmp <- data.matrix(score_final_tmp)
  #rownames(score_final_tmp) <- colnames(feature_data)[i]
  #colnames(score_final_tmp) <- "score"
  score_final <- rbind(score_final, score_final_tmp)
  
}

# RF OS cutoff

score_final_CGGA <- data.frame(score_final)
#cutoff_RF_OS <- median(score_final_CGGA$score)
#print(score_final_CGGA)
score_final_CGGA$group <- 1
score_final_CGGA$group[which(score_final_CGGA$score > cutoff_RF_OS)] <- 2
#score_final_CGGA$group[which(score_final_CGGA$score > median(score_final_CGGA$score))] <- 2
#score_final_CGGA$Censor <- t(CGGA["Censor",rownames(score_final_CGGA)])
#score_final_CGGA$OS <- t(CGGA["OS",rownames(score_final_CGGA)])
print(score_final_CGGA$group)
# not use the previous clinical data, use new tmp <- as.data.frame(t(CGGA))


#保存数据
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

