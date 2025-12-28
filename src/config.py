import os
import torch

# ================= 基础路径 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

DOWNLOAD_BASE = os.path.join(BASE_DIR, "test_downloads")
# 新增 Ground Truth 路径
GT_FILE = os.path.join(DOWNLOAD_BASE, "ground_truth.json")

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ================= 硬件配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# ================= 模型配置 =================
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" 
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
LLM_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# ================= 标签定义 (含 NLP) =================
LABEL_DEFINITIONS = {
    "Computer Vision": (
        "Research focused on enabling computers to interpret and understand visual information from the world. "
        "Key topics include: Image Classification (ResNet, EfficientNet), Object Detection (YOLO, Faster R-CNN, SSD), "
        "Image Segmentation (SAM, Mask R-CNN, U-Net), Face Recognition, Pose Estimation, 3D Reconstruction (NeRF, Gaussian Splatting), "
        "Video Analysis, Optical Flow, Medical Imaging, and architectures like Vision Transformers (ViT) and Convolutional Neural Networks (CNNs). "
        "NOTE: Pure text-based Attention papers belong to NLP."
    ),

    "Natural Language Processing": (
        "Research concerning the interaction between computers and human language, focusing on understanding, interpreting, and manipulating text. "
        "Key topics include: Text Classification, Named Entity Recognition (NER), Sentiment Analysis, Machine Translation, "
        "Syntactic Parsing, Question Answering systems (extractive), Word Embeddings (Word2Vec, GloVe), and pre-LLM transformer models "
        "like BERT, RoBERTa, and T5 focused on specific downstream tasks rather than general generation."
    ),

    "Large Language Models": (
        "Research specifically dedicated to massive-scale generative models, training methodologies, and application strategies. "
        "Key topics include: Foundation Models (GPT-4, LLaMA, Claude, Qwen, Mistral), Instruction Tuning (InstructGPT), "
        "Reinforcement Learning from Human Feedback (RLHF), Prompt Engineering (Chain-of-Thought, Tree-of-Thought), "
        "Retrieval-Augmented Generation (RAG), Parameter-Efficient Fine-Tuning (LoRA, QLoRA), Context Window optimization, and Chatbot development."
    ),

    "Reinforcement Learning": (
        "Research on computational agents learning to make decisions by performing actions in an environment to maximize cumulative reward. "
        "Key topics include: Deep Reinforcement Learning (Deep RL), Policy Gradients, Q-Learning (DQN), Proximal Policy Optimization (PPO), "
        "Actor-Critic methods (A3C, SAC), Multi-Agent Reinforcement Learning (MARL), Robotics Control, Sim-to-Real transfer, "
        "Game playing agents (AlphaGo, AlphaZero), and Offline RL. "
        "NOTE: Do NOT classify Biology or Neuroscience papers here."
    ),

    "AI Generated Content": (
        "Research focusing on the creation of new data instances (images, audio, video, 3D assets) that resemble original data. "
        "Key topics include: Diffusion Models (Stable Diffusion, Midjourney, DALL-E), Generative Adversarial Networks (GANs), "
        "Variational Autoencoders (VAEs), Text-to-Image, Text-to-Video (Sora), Image Inpainting, Super-resolution, "
        "Voice Synthesis (TTS), and Neural Style Transfer."
    ),

    "Biology": (
        "Research involving the study of living organisms, molecular structures, and biological data analysis. "
        "Key topics include: Genomics, Bioinformatics, Computational Biology, Protein Structure Prediction (AlphaFold, Rosetta), "
        "Single-cell RNA sequencing (scRNA-seq) data analysis, Spatial Transcriptomics, Drug Discovery, Molecular Dynamics, "
        "CRISPR/Gene Editing, and Systems Biology."
    ),

    "Finance": (
        "Research related to economic systems, financial markets, and quantitative trading strategies. "
        "Key topics include: Algorithmic Trading, High-Frequency Trading (HFT), Limit Order Books (LOB), Market Microstructure, "
        "Time Series Forecasting for stock prices, Risk Management, Portfolio Optimization, Fintech, Cryptocurrency analysis, "
        "and Volatility Modeling (GARCH, Stochastic Volatility)."
    ),

    "Neuroscience": (
        "Research on the nervous system, brain function, and cognitive processes. "
        "Key topics include: Brain-Computer Interfaces (BCI), Functional MRI (fMRI) analysis, EEG signal processing, "
        "Spiking Neural Networks (SNNs), Cognitive Science, Neurobiology, Neural Encoding/Decoding, "
        "study of biological neural circuits, and research into neurodegenerative diseases."
    ),

    "Physics": (
        "Research in the fundamental sciences of matter and energy. "
        "Key topics include: Astrophysics, Quantum Mechanics, Quantum Computing (hardware/theory), Particle Physics (High Energy Physics), "
        "Condensed Matter, Thermodynamics, Fluid Dynamics, General Relativity, and Physics-Informed Neural Networks (PINNs) applied to physical simulations. "
        "NOTE: Financial stochastic models belong to Finance."
    )
}