#!/usr/bin/env Rscript
# 虚拟筛选结果可视化脚本
# 用于分析和展示筛选结果的分布情况

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# 读取筛选结果
read_screening_results <- function() {
  # 假设筛选结果存储在results/models/virtual_screening目录下
  result_files <- list.files("../results/models/virtual_screening", pattern="screening_results_*.csv", full.names=TRUE)
  
  if (length(result_files) == 0) {
    # 如果没有找到CSV文件，创建模拟数据
    cat("没有找到筛选结果文件，使用模拟数据\n")
    return(create_mock_screening_data())
  }
  
  # 读取最新的筛选结果文件
  latest_file <- result_files[which.max(file.mtime(result_files))]
  cat(paste("读取筛选结果文件:", latest_file, "\n"))
  
  # 读取数据
  results <- read.csv(latest_file)
  return(results)
}

# 创建模拟数据（如果没有实际数据）
create_mock_screening_data <- function() {
  # 创建1000个化合物的模拟数据
  n_compounds <- 1000
  data.frame(
    compound_id = paste("CMPD", 1:n_compounds, sep="_"),
    canonical_smiles = paste("SMILES_", 1:n_compounds, sep=""),
    RandomForest_prediction = sample(c(0, 1), n_compounds, replace=TRUE, prob=c(0.7, 0.3)),
    XGBoost_prediction = sample(c(0, 1), n_compounds, replace=TRUE, prob=c(0.6, 0.4)),
    SVM_prediction = sample(c(0, 1), n_compounds, replace=TRUE, prob=c(0.65, 0.35)),
    average_probability = runif(n_compounds, 0, 1),
    is_active = sample(c(0, 1), n_compounds, replace=TRUE, prob=c(0.7, 0.3))
  )
}

# 绘制活性化合物概率分布
plot_probability_distribution <- function(data) {
  ggplot(data, aes(x = average_probability)) +
    geom_histogram(bins = 30, fill = "skyblue", alpha = 0.7) +
    geom_vline(xintercept = 0.5, color = "red", linetype = "dashed", size = 1) +
    labs(title = "化合物活性概率分布",
         x = "平均活性概率",
         y = "化合物数量") +
    theme_minimal()
}

# 绘制模型预测一致性
plot_model_consistency <- function(data) {
  # 计算模型预测的一致性
  data <- data %>%
    mutate(consensus = RandomForest_prediction + XGBoost_prediction + SVM_prediction)
  
  ggplot(data, aes(x = factor(consensus))) +
    geom_bar(fill = "green", alpha = 0.7) +
    labs(title = "模型预测一致性",
         x = "一致预测的模型数量",
         y = "化合物数量") +
    theme_minimal()
}

# 绘制活性化合物数量统计
plot_active_compounds <- function(data) {
  # 计算各模型预测的活性化合物数量
  active_counts <- data %>%
    summarize(
      RandomForest = sum(RandomForest_prediction),
      XGBoost = sum(XGBoost_prediction),
      SVM = sum(SVM_prediction),
      Consensus = sum(is_active)
    ) %>%
    pivot_longer(everything(), names_to = "model", values_to = "count")
  
  ggplot(active_counts, aes(x = model, y = count, fill = model)) +
    geom_bar(stat = "identity") +
    labs(title = "各模型预测的活性化合物数量",
         x = "模型",
         y = "活性化合物数量") +
    theme_minimal()
}

# 主函数
main <- function() {
  # 读取筛选结果
  screening_data <- read_screening_results()
  
  # 打印数据摘要
  print("筛选结果摘要:")
  print(summary(screening_data))
  
  # 创建输出目录
  dir.create("../results/figures", recursive = TRUE, showWarnings = FALSE)
  
  # 绘制并保存概率分布图
  prob_plot <- plot_probability_distribution(screening_data)
  ggsave("../results/figures/probability_distribution.png", prob_plot, width = 10, height = 6)
  cat("概率分布图已保存到 ../results/figures/probability_distribution.png\n")
  
  # 绘制并保存模型一致性图
  consistency_plot <- plot_model_consistency(screening_data)
  ggsave("../results/figures/model_consistency.png", consistency_plot, width = 10, height = 6)
  cat("模型一致性图已保存到 ../results/figures/model_consistency.png\n")
  
  # 绘制并保存活性化合物数量图
  active_plot <- plot_active_compounds(screening_data)
  ggsave("../results/figures/active_compounds_count.png", active_plot, width = 10, height = 6)
  cat("活性化合物数量图已保存到 ../results/figures/active_compounds_count.png\n")
  
  # 显示图表
  print(prob_plot)
  print(consistency_plot)
  print(active_plot)
}

# 运行主函数
if (!interactive()) {
  main()
}
