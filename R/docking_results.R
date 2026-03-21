#!/usr/bin/env Rscript
# 虚拟对接结果可视化脚本
# 用于分析和展示对接结果的分布情况

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# 读取对接结果
read_docking_results <- function() {
  # 假设对接结果存储在results/models/docking目录下
  result_files <- list.files("../results/models/docking", pattern="docking_results_*.csv", full.names=TRUE)
  
  if (length(result_files) == 0) {
    # 如果没有找到CSV文件，创建模拟数据
    cat("没有找到对接结果文件，使用模拟数据\n")
    return(create_mock_docking_data())
  }
  
  # 读取最新的对接结果文件
  latest_file <- result_files[which.max(file.mtime(result_files))]
  cat(paste("读取对接结果文件:", latest_file, "\n"))
  
  # 读取数据
  results <- read.csv(latest_file)
  return(results)
}

# 创建模拟数据（如果没有实际数据）
create_mock_docking_data <- function() {
  # 创建100个化合物的模拟数据
  n_compounds <- 100
  data.frame(
    compound_id = paste("CMPD", 1:n_compounds, sep="_"),
    smiles = paste("SMILES_", 1:n_compounds, sep=""),
    probability = runif(n_compounds, 0.4, 1.0),
    affinity = runif(n_compounds, -10, -1),
    rmsd_lb = runif(n_compounds, 0, 2),
    rmsd_ub = runif(n_compounds, 1, 3)
  )
}

# 绘制对接亲和力分布
plot_affinity_distribution <- function(data) {
  ggplot(data, aes(x = affinity)) +
    geom_histogram(bins = 30, fill = "purple", alpha = 0.7) +
    geom_vline(xintercept = -5, color = "red", linetype = "dashed", size = 1) +
    labs(title = "对接亲和力分布",
         x = "亲和力 (kcal/mol)",
         y = "化合物数量") +
    theme_minimal()
}

# 绘制亲和力与活性概率的关系
plot_affinity_vs_probability <- function(data) {
  ggplot(data, aes(x = probability, y = affinity, color = affinity)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_gradient(low = "red", high = "blue") +
    labs(title = "亲和力与活性概率的关系",
         x = "活性概率",
         y = "亲和力 (kcal/mol)") +
    theme_minimal()
}

# 绘制Top 20化合物的亲和力排序
plot_top_compounds <- function(data) {
  # 按亲和力排序，选择Top 20
  top_compounds <- data %>%
    arrange(affinity) %>%
    head(20)
  
  ggplot(top_compounds, aes(x = reorder(compound_id, affinity), y = affinity)) +
    geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
    coord_flip() +
    labs(title = "Top 20 化合物亲和力排序",
         x = "化合物ID",
         y = "亲和力 (kcal/mol)") +
    theme_minimal()
}

# 主函数
main <- function() {
  # 读取对接结果
  docking_data <- read_docking_results()
  
  # 打印数据摘要
  print("对接结果摘要:")
  print(summary(docking_data))
  
  # 创建输出目录
  dir.create("../results/figures", recursive = TRUE, showWarnings = FALSE)
  
  # 绘制并保存亲和力分布图
  affinity_plot <- plot_affinity_distribution(docking_data)
  ggsave("../results/figures/affinity_distribution.png", affinity_plot, width = 10, height = 6)
  cat("亲和力分布图已保存到 ../results/figures/affinity_distribution.png\n")
  
  # 绘制并保存亲和力与概率关系图
  affinity_prob_plot <- plot_affinity_vs_probability(docking_data)
  ggsave("../results/figures/affinity_vs_probability.png", affinity_prob_plot, width = 10, height = 6)
  cat("亲和力与概率关系图已保存到 ../results/figures/affinity_vs_probability.png\n")
  
  # 绘制并保存Top 20化合物图
  top_compounds_plot <- plot_top_compounds(docking_data)
  ggsave("../results/figures/top_compounds.png", top_compounds_plot, width = 10, height = 8)
  cat("Top 20化合物图已保存到 ../results/figures/top_compounds.png\n")
  
  # 显示图表
  print(affinity_plot)
  print(affinity_prob_plot)
  print(top_compounds_plot)
}

# 运行主函数
if (!interactive()) {
  main()
}
