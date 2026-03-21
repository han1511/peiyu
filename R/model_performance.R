#!/usr/bin/env Rscript
# 模型性能可视化脚本
# 用于比较不同模型的性能指标

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(jsonlite)

# 读取模型性能数据
read_model_performance <- function() {
  # 假设模型性能数据存储在results/models目录下
  model_files <- list.files("../results/models", pattern="*_results.json", full.names=TRUE)
  
  if (length(model_files) == 0) {
    # 如果没有找到JSON文件，创建模拟数据
    cat("没有找到模型性能文件，使用模拟数据\n")
    return(create_mock_model_data())
  }
  
  # 读取并合并模型性能数据
  performance_data <- data.frame()
  
  for (file in model_files) {
    model_name <- gsub("_results.json", "", basename(file))
    data <- fromJSON(file)
    
    # 提取性能指标
    metrics <- data.frame(
      model = model_name,
      accuracy = data$accuracy,
      precision = data$precision,
      recall = data$recall,
      f1_score = data$f1_score,
      auc = data$auc
    )
    
    performance_data <- rbind(performance_data, metrics)
  }
  
  return(performance_data)
}

# 创建模拟数据（如果没有实际数据）
create_mock_model_data <- function() {
  data.frame(
    model = c("RandomForest", "XGBoost", "SVM"),
    accuracy = c(0.98, 0.99, 0.97),
    precision = c(0.99, 1.0, 0.98),
    recall = c(0.97, 0.98, 0.96),
    f1_score = c(0.98, 0.99, 0.97),
    auc = c(0.99, 1.0, 0.98)
  )
}

# 绘制模型性能雷达图
plot_model_radar <- function(data) {
  # 转换数据为长格式
  long_data <- data %>%
    select(-model) %>%
    mutate(model = data$model) %>%
    pivot_longer(cols = -model, names_to = "metric", values_to = "value")
  
  # 绘制雷达图
  ggplot(long_data, aes(x = metric, y = value, color = model, group = model)) +
    geom_polygon(fill = NA, size = 1) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    scale_y_continuous(limits = c(0.9, 1.0)) +
    labs(title = "模型性能雷达图",
         x = "性能指标",
         y = "得分") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# 绘制模型性能条形图
plot_model_bar <- function(data) {
  # 转换数据为长格式
  long_data <- data %>%
    select(-model) %>%
    mutate(model = data$model) %>%
    pivot_longer(cols = -model, names_to = "metric", values_to = "value")
  
  # 绘制条形图
  ggplot(long_data, aes(x = model, y = value, fill = metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_y_continuous(limits = c(0.9, 1.0)) +
    labs(title = "模型性能比较",
         x = "模型",
         y = "得分") +
    theme_minimal()
}

# 主函数
main <- function() {
  # 读取模型性能数据
  performance_data <- read_model_performance()
  
  # 打印数据
  print("模型性能数据:")
  print(performance_data)
  
  # 创建输出目录
  dir.create("../results/figures", recursive = TRUE, showWarnings = FALSE)
  
  # 绘制并保存雷达图
  radar_plot <- plot_model_radar(performance_data)
  ggsave("../results/figures/model_performance_radar.png", radar_plot, width = 10, height = 8)
  cat("雷达图已保存到 ../results/figures/model_performance_radar.png\n")
  
  # 绘制并保存条形图
  bar_plot <- plot_model_bar(performance_data)
  ggsave("../results/figures/model_performance_bar.png", bar_plot, width = 10, height = 8)
  cat("条形图已保存到 ../results/figures/model_performance_bar.png\n")
  
  # 显示图表
  print(radar_plot)
  print(bar_plot)
}

# 运行主函数
if (!interactive()) {
  main()
}
