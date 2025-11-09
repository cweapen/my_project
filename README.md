# Python 实验项目

本项目为实验项目，供我这个初学者记录学习所用，有诸多不妥之处。包含训练和测试代码，支持复现实验。使用 Git 管理，并可在 GitHub 上开源共享。

---

## 项目结构
```text
src/
├── data/
│   ├── data_set.py
│   ├── dataset.py
│   └── download_data.py
├── model/
│   ├── __init__.py
│   ├── model.py
│   ├── model_1.py
│   └── model_2.py
├── results/               # 训练结果、曲线图
├── scripts/
│   ├── test_forward.py
│   ├── train_ablation.py
│   ├── train_challenge.py
│   ├── train_full.py
│   └── train_small.py
└── utils/
    └── mask.py
.gitignore                 # Git 忽略文件
README.md                  # 项目说明
