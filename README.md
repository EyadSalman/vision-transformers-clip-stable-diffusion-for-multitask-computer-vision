# 🧠 Vision Transformers + CLIP + Stable Diffusion for Multitask Computer Vision

This project combines **Vision Transformers (ViT)**, **CLIP embeddings**, and **Stable Diffusion** to tackle multitask computer vision challenges—primarily focused on **face anti-spoofing detection** using the CelebA-Spoof dataset. It sets the stage for future integration with generative and multimodal tasks.

---

## 🎯 Objective

Demonstrate a scalable computer vision pipeline by:
- Training a **ViT-based classifier** for real vs spoof face detection
- Preparing for multimodal extension with **CLIP embeddings**
- Exploring **Stable Diffusion** for visual insights or data augmentation

Evaluation focuses on:
- Model Accuracy
- Training Efficiency
- Parameter Scalability
- Visual Interpretability

---

## 📊 Dataset

- **CelebA-Spoof Face Anti-Spoofing Dataset**
- **Total Samples**: ~15,000  
  - Subset used: 20% for faster experimentation
- **Split**:
  - 80% Training
  - 20% Testing
- **Source**: [Hugging Face - nguyenkhoa/celeba-spoof-for-face-antispoofing-test](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test)

---

## 🧠 Core Components

| Component         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Vision Transformer** | Backbone classifier using `vit_base_patch16_224` from `timm`                  |
| **CLIP (Planned)**     | Multimodal feature embedding for semantic alignment (extendable)        |
| **Stable Diffusion**   | Diffusers pipeline for visualization and potential data augmentation     |
| **Feature Extractor**  | `AutoFeatureExtractor` from Hugging Face for image preprocessing         |

---

## ⚙️ Project Structure

```bash
.
├── vit_clip_stable_diffusion.ipynb        # Main Jupyter notebook
├── README.md                              # Project documentation
├── train_small_dataset/                   # Preprocessed training subset
├── test_small_dataset/                    # Preprocessed testing subset
├── /models                                # (Optional) Saved models
├── /visualizations                        # (Optional) Output from Stable Diffusion
└── requirements.txt                       # Python dependencies
