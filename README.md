# Local Multimodal AI Agent (本地多模态 AI 智能体)

LocalAI_Agent/
├── data/                  # 存放向量数据库和分类后的文件
├── test_downloads/        # 存放下载的原始测试数据
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── config.py          # 配置项
│   ├── database.py        # ChromaDB 封装
│   ├── models.py          # 模型加载与推理
│   └── utils.py           # 文件操作, PDF解析等工具函数
├── main.py                # 主入口程序
├── download_new_data.py               # 批量数据下载与 Ground Truth 生成脚本
├── download_single_test_data.py       # 单个数据测试脚本
├── requirements.txt               
├── bash_list.txt          # 项目演示所用命令
└── README.md              # 项目说明文档

## 📖 项目简介

**Local Multimodal AI Agent** 是一个基于本地硬件（支持 NVIDIA GPU）运行的隐私优先、高性能多模态知识库管理系统。

该项目旨在解决个人或科研场景下大量文献（PDF）和图片素材的整理与检索难题。它不依赖任何云端 API（如 OpenAI API），完全在本地离线运行，确保数据隐私安全。

### ✨ 核心功能

* **智能文献分类 (Hybrid Classification)**: 
    * 采用 **"规则引擎 + 大模型 (LLM)"** 的混合专家策略 (Mixture of Experts approach)。
    * 优先使用基于文件名和内容的**强规则匹配**（覆盖率高，速度快）。
    * 对于长尾复杂样本，使用本地部署的 **Qwen2.5-3B** 模型进行深度语义理解和逻辑推理分类。
    * 支持自定义分类体系（如 Physics, CV, NLP, AIGC 等）。
* **非破坏性整理**: 自动将混乱的下载文件**复制**并归档到结构化的分类文件夹中，保留原始文件不动。
* **多模态语义检索 (RAG & Text-to-Image Search)**:
    * **以文搜文**: 基于 `Sentence-Transformers` 实现对论文内容的深度语义搜索。
    * **以文搜图**: 基于 `OpenAI CLIP` 模型，支持用自然语言搜索本地图片库（如 "搜索一张猫在海边的照片"）。
* **本地向量数据库**: 内置 `ChromaDB`，无需安装额外的数据库服务，开箱即用，支持持久化存储。
* **准确率评估**: 内置 Ground Truth 比对机制，可自动生成详细的分类准确率报告。

---

## 🛠️ 技术选型

本项目完全基于开源生态构建：

| 组件 | 模型/工具 | 说明 |
| :--- | :--- | :--- |
| **LLM (推理核心)** | `Qwen/Qwen3-4B-Instruct-2507` | 阿里通义千问开源小模型，平衡了显存占用与推理能力。 |
| **Text Embedding** | `sentence-transformers/all-mpnet-base-v2` | 目前 SOTA 级别的开源句向量模型，用于文档检索。 |
| **Image Embedding** | `openai/clip-vit-large-patch14` | 经典的图文对齐模型，用于零样本图像分类和检索。 |
| **Vector DB** | `ChromaDB` | 轻量级、嵌入式的向量数据库。 |
| **PDF Processing** | `PyMuPDF (fitz)` | 高速 PDF 文本提取工具。 |
| **Infrastructure** | `PyTorch` + `CUDA` | 深度学习计算后端。 |

---

## 💻 环境配置与安装

### 1. 基础环境
建议使用 Python 3.10+ 和 Conda 环境。

```bash
# 创建并激活环境
conda create -n ai_agent python=3.10
conda activate ai_agent

# 安装 PyTorch (请根据你的 CUDA 版本调整，以下为 CUDA 12.1 示例)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. 安装项目依赖
```bash
git clone [https://github.com/YodelYang/Local-Multimodal-AI-Agent.git](https://github.com/YodelYang/Local-Multimodal-AI-Agent.git)
cd LocalAI_Agent
pip install -r requirements.txt
```

---

## 🚀 使用说明

### 1. 准备测试数据
项目提供了一键生成测试数据的脚本，会自动下载 arXiv 论文和 Unsplash 图片，并生成 Ground Truth 标签。

```bash
python download_test_data.py
```
> *下载完成后，数据位于 `test_downloads/raw_pdfs` 目录。*

### 2. 执行智能整理 (核心功能)
该命令会自动扫描指定目录的 PDF，进行分类、复制归档、生成向量索引，并输出准确率报告。

```bash
python main.py organize_folder "./test_downloads/raw_pdfs" \
    --topics "Reinforcement Learning,Natural Language Processing,Large Language Models,Computer Vision,AI Generated Content,Physics,Biology,Finance,Neuroscience"
```

**运行结果示例：**
```text
📊 CLASSIFICATION REPORT
============================================================
📂 Category Distribution:
   - Reinforcement Learning         : 7
   - Computer Vision                : 7
   ...
✅ Total Verified: 36
🎯 Accuracy:       100.00% (36/36)
🎉 Perfect Score! All classifications match ground truth.
```

### 3. 单个文献处理 (Single File Mode)
除了批量整理文件夹，系统也支持针对**单篇 PDF 文档**进行精准分类、归档和索引。

**第一步：获取单篇测试数据**
运行以下脚本，下载特定的测试论文（如 GPT-4 Technical Report）到独立目录：

```bash
python download_single_test_data.py
```
**第二步：添加并分类**
指定文件路径和候选分类列表进行处理：

```bash
python main.py add_paper "./test_downloads/single_test/GPT-4 Technical Report.pdf" \
    --topics "Reinforcement Learning,Natural Language Processing,Large Language Models,Computer Vision,AI Generated Content,Physics,Biology,Finance,Neuroscience"
```

**运行结果示例：**
```text
⏳ Loading Models on cuda...
✅ All Models Loaded Successfully.
✅ GPT-4 Technical Report.pdf             -> [Large Language Models]
```
注：处理完成后，该文件会被复制到 data/papers/Large Language Models/ 目录下，并建立向量索引。


### 4. 建立图片索引
对图片文件夹进行语义索引（同样会复制并归档）。

```bash
python main.py index_images "./test_downloads/raw_images"
```

### 5. 语义搜索测试

**搜索论文：**
```bash
python main.py search_paper "How does the attention mechanism work?"
```

**以文搜图：**
```bash
python main.py search_image "A city view"
```

---

## ⚙️ 进阶配置

所有核心配置均位于 `src/config.py` 和 `src/models.py`。

* **修改分类定义**: 在 `src/config.py` 中的 `LABEL_DEFINITIONS` 修改 Prompt 定义。
* **添加强规则**: 在 `src/models.py` 中的 `KEYWORD_RULES` 字典中添加关键词映射，可强制纠正 LLM 的分类错误。
