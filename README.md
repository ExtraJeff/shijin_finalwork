# Wordle游戏预测项目

## 项目概述

本项目旨在使用多种深度学习模型预测Wordle游戏的结果，包括尝试次数预测和成功率预测。通过分析玩家的历史尝试序列，构建了LSTM、BiLSTM、LSTM-Attention和Transformer四种模型，并进行了全面的性能比较和分析。

## 项目结构

### 根目录文件
```
final_homework/
├── README.md                                # 项目说明文档
├── requirements.txt                         # 项目依赖包列表
├── dashboard.py                             # 交互式仪表盘脚本
├── Wordle_Dashboard_Feature_Description.md  # 仪表盘功能说明
├── get_wandb_links.py                       # 获取WandB链接脚本
├── test_wandb.py                            # WandB测试脚本
├── wandb_links.txt                          # WandB链接记录
└── wordle_lstm_project/                     # 主项目目录
```

### 主项目目录结构
```
wordle_lstm_project/
├── 01_raw_data/             # 原始数据
│   └── wordle_games.csv     # 原始Wordle游戏数据
├── 02_data_preprocessing/   # 数据预处理
│   ├── output/
│   │   └── wordle_preprocessed.csv  # 预处理后的数据
│   └── data_preprocessing.py        # 数据预处理脚本
├── 03_feature_engineering/  # 特征工程
│   ├── output/
│   │   └── wordle_with_player_features.csv  # 带有玩家特征的数据
│   └── player_statistics.py        # 玩家特征计算脚本
├── 04_dataset_construction/ # 数据集构建
│   ├── output/
│   │   ├── X_feat.npy       # 特征数据
│   │   ├── X_seq.npy        # 序列数据
│   │   └── y.npy            # 标签数据
│   └── build_lstm_dataset.py        # 数据集构建脚本
├── 05_models/               # 模型目录
│   ├── bilstm/              # BiLSTM模型
│   │   ├── output/          # 模型输出
│   │   │   ├── bilstm_error_by_activity.png   # 玩家活跃度与误差关系图
│   │   │   ├── bilstm_main_model.keras         # 保存的模型文件
│   │   │   ├── bilstm_metrics_summary.png      # 模型指标摘要图
│   │   │   ├── bilstm_prediction_analysis.png  # 预测分布与误差图
│   │   │   └── bilstm_training_curves.png      # 训练曲线
│   │   ├── wandb/           # WandB运行记录
│   │   ├── bilstm_model.py  # 模型定义
│   │   └── train_bilstm.py  # 训练脚本
│   ├── lstm/                # LSTM模型
│   │   ├── output/          # 模型输出
│   │   │   ├── lstm_error_by_activity.png   # 玩家活跃度与误差关系图
│   │   │   ├── lstm_main_model.keras         # 保存的模型文件
│   │   │   ├── lstm_metrics_summary.png      # 模型指标摘要图
│   │   │   ├── lstm_prediction_analysis.png  # 预测分布与误差图
│   │   │   └── lstm_training_curves.png      # 训练曲线
│   │   ├── wandb/           # WandB运行记录
│   │   ├── lstm_model.py    # 模型定义
│   │   ├── train_lstm.py    # 训练脚本
│   │   └── update_prediction_plot.py        # 更新预测图脚本
│   ├── lstm_attention/      # LSTM-Attention模型
│   │   ├── output/          # 模型输出
│   │   │   ├── lstm_attention_error_by_activity.png   # 玩家活跃度与误差关系图
│   │   │   ├── lstm_attention_model.keras         # 保存的模型文件
│   │   │   ├── lstm_attention_metrics_summary.png      # 模型指标摘要图
│   │   │   ├── lstm_attention_prediction_analysis.png  # 预测分布与误差图
│   │   │   └── lstm_attention_training_curves.png      # 训练曲线
│   │   ├── wandb/           # WandB运行记录
│   │   ├── lstm_attention_model.py  # 模型定义
│   │   └── train_lstm_attention.py  # 训练脚本
│   └── transformer/         # Transformer模型
│       ├── output/          # 模型输出
│       │   ├── transformer_error_by_activity.png   # 玩家活跃度与误差关系图
│       │   ├── transformer_metrics_summary.png      # 模型指标摘要图
│       │   ├── transformer_model.keras         # 保存的模型文件
│       │   ├── transformer_prediction_analysis.png  # 预测分布与误差图
│       │   └── transformer_training_curves.png      # 训练曲线
│       ├── train_transformer.py  # 训练脚本
│       └── transformer_model.py  # 模型定义
├── 06_analysis/             # 分析脚本
│   ├── output/              # 分析结果
│   │   ├── error_by_activity_all_models.png    # 所有模型的玩家活跃度与误差关系图
│   │   ├── feature_error_correlation_heatmaps.png  # 特征与误差相关性热力图
│   │   ├── lstm_confusion_matrix_roc.png       # LSTM模型混淆矩阵和ROC曲线
│   │   ├── model_comparison_*.png              # 各种模型比较图
│   │   └── model_performance_boxplot.png       # 模型性能箱线图
│   ├── analyze_boxplot_data.py                 # 箱线图数据分析脚本
│   ├── generate_all_models_comparison.py       # 生成所有模型比较图脚本
│   ├── generate_confusion_roc.py               # 生成混淆矩阵和ROC曲线脚本
│   └── model_comparison_analysis.py            # 模型比较分析脚本
├── 07_results/              # 模型评估结果
│   └── model_evaluation_results.csv            # 模型评估结果表
├── 08_attention_analysis/   # 注意力机制分析
│   ├── output/              # 注意力分析结果
│   │   ├── attention_sample_*.png              # 注意力权重示例图
│   │   ├── attention_weights_bar.png           # 注意力权重条形图
│   │   ├── attention_weights_heatmap.png       # 注意力权重热力图
│   │   └── attention_weights_line.png          # 注意力权重折线图
│   ├── analyze_attention_weights.py            # 注意力权重分析脚本
│   └── simple_attention_analysis.py            # 简单注意力分析脚本
├── utils/                   # 工具函数
│   └── vis_utils.py         # 可视化工具函数
└── 研究报告.md              # 项目研究报告
```

## 主要功能

1. **数据处理**：从原始Wordle游戏数据中提取有用信息，进行清洗和格式化
2. **特征工程**：计算玩家统计特征，构建适合模型训练的特征集
3. **模型训练**：实现并训练四种深度学习模型
4. **模型评估**：对模型进行全面的性能评估，包括回归和分类指标
5. **结果分析**：生成各种可视化图表，分析模型性能和行为
6. **注意力机制**：对LSTM-Attention模型进行注意力权重分析，增强模型可解释性
7. **交互式仪表盘**：使用Streamlit构建可视化仪表盘，展示模型结果

## 技术栈

- **Python 3.10+**
- **深度学习框架**：TensorFlow 2.x
- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **模型评估**：scikit-learn
- **实验跟踪**：Weights & Biases (WandB)
- **工具库**：自定义可视化工具函数(vis_utils.py)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

1. **数据预处理**
   ```bash
   cd wordle_lstm_project/02_data_preprocessing
   python data_preprocessing.py
   ```

2. **特征工程**
   ```bash
   cd ../03_feature_engineering
   python player_statistics.py
   ```

3. **构建数据集**
   ```bash
   cd ../04_dataset_construction
   python build_lstm_dataset.py
   ```

4. **训练模型**
   - **LSTM模型**
     ```bash
     cd ../05_models/lstm
     python train_lstm.py
     ```
   - **BiLSTM模型**
     ```bash
     cd ../05_models/bilstm
     python train_bilstm.py
     ```
   - **LSTM-Attention模型**
     ```bash
     cd ../05_models/lstm_attention
     python train_lstm_attention.py
     ```
   - **Transformer模型**
     ```bash
     cd ../05_models/transformer
     python train_transformer.py
     ```

5. **运行分析脚本**
   ```bash
   cd ../../06_analysis
   python model_comparison_analysis.py
   ```

6. **分析注意力权重**（仅LSTM-Attention模型）
   ```bash
   cd ../08_attention_analysis
   python analyze_attention_weights.py
   ```

## 模型比较

| 模型 | MAE | RMSE | 准确率 | F1分数 | AUC |
|------|-----|------|--------|--------|-----|
| LSTM | 1.0772 | 1.3171 | 0.9520 | 0.9754 | 0.7503 |
| BiLSTM | 1.0708 | 1.3133 | 0.9521 | 0.9754 | 0.7801 |
| LSTM-Attention | 1.0719 | 1.3135 | 0.9522 | 0.9755 | 0.7684 |
| Transformer | 1.0781 | 1.3169 | 0.9521 | 0.9754 | 0.7664 |

## 关键发现

1. **序列建模有效性**：所有模型都能有效捕捉玩家行为的时序模式
2. **双向信息优势**：BiLSTM模型在回归和分类任务中表现最佳
3. **注意力机制洞察**：模型对中间尝试步骤（Step 3）分配最高权重
4. **复杂度与性能**：在短序列场景下，Transformer模型并未显著优于基于循环结构的模型

## 仪表盘功能

- 模型性能比较可视化
- 模型详情展示
- 最佳模型推荐
- 模型特定可视化（训练曲线、指标摘要、预测分析、误差分布）
- 模型比较可视化（全指标比较图、雷达图）
- 模型架构和训练信息
- 注意力机制分析
- 结论和建议

## 使用说明

### 数据格式

原始数据包含以下字段：
- game_id：游戏ID
- date：游戏日期
- player_id：玩家ID
- target_word：目标单词
- attempts：尝试次数
- result：结果（成功/失败）
- attempt1_guess：第一次尝试的猜测
- attempt1_result：第一次尝试的结果
- ...
- attempt6_guess：第六次尝试的猜测
- attempt6_result：第六次尝试的结果

### 模型训练参数

- 批大小：128
- 训练轮数：30（带早停机制）
- 优化器：AdamW
- 学习率：0.0003
- 损失函数：回归任务使用MAE，分类任务使用二元交叉熵

## 结果解释

- **回归任务**：预测玩家完成游戏所需的尝试次数
- **分类任务**：预测玩家是否成功完成游戏
- **注意力权重**：显示模型在预测时对不同尝试步骤的关注程度

## 未来改进方向

1. 引入更详细的玩家行为特征
2. 建模玩家跨游戏的学习演化过程
3. 解决类别不平衡问题
4. 设计针对Wordle规则的特定模型结构
5. 探索强化学习和图结构建模

## 许可证

本项目采用MIT许可证。

## 作者

231820009 刘宇轩