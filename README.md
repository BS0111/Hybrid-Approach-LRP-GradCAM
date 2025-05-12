# 🍇 Visualizing Explanations of DCNN-Based Plant Leaf Disease Detection Using Combined Grad-CAM and LRP

This project aims to classify grape leaf diseases using a fine-tuned VGG16 model, enhanced with state-of-the-art explainability techniques like **Grad-CAM++**, **Layer-wise Relevance Propagation (LRP)**, and their **combined visualizations**. Furthermore, the reliability of these explanations is quantitatively assessed using **infidelity metrics**.

---

## 📌 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Explainability Methods](#explainability-methods)
- [Infidelity Evaluation](#infidelity-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## 🧠 About the Project

Accurate detection and interpretation of plant diseases are essential for smart agriculture. This repository provides a robust deep learning pipeline to classify grape leaf diseases with built-in **visual explanations** and **faithfulness evaluation**.

Key Features:
- Fine-tuned **VGG16** model on grape leaf dataset.
- Visualization using **Grad-CAM++**, **LRP**, and **combined heatmaps**.
- Faithfulness evaluation using **Infidelity metrics** from Captum.
- Batch-wise predictions and explanation saving for large-scale analysis.

---

## 🌿 Dataset

The dataset used is the **Grape Leaf Disease** dataset, organized as follows:
```
grape_dataset/
├── train/
│   ├── Black Rot/
│   ├── Esca (Black Measles)/
│   ├── Leaf Blight/
│   └── Healthy/
├── test/
│   └── (same class structure)
```

Each class contains images of grape leaves affected by specific diseases or healthy leaves.

---

## 🏗️ Model Architecture

- Base model: **VGG16** pretrained on ImageNet.
- Modified last layer for 4-class classification.
- Trained using PyTorch on the grape dataset.

```python
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 4)
```

---

## 🔍 Explainability Methods

1. **Grad-CAM++**  
   Uses gradients flowing into the last convolutional layer to highlight important regions in the image.

2. **Layer-wise Relevance Propagation (LRP)**  
   Distributes the prediction score backward through the network to attribute relevance scores to input pixels.

3. **Combined Heatmap**  
   A fusion of Grad-CAM++ and LRP to produce richer and more interpretable visualizations.

---

## 📊 Infidelity Evaluation

To evaluate the **faithfulness** of explanations, we use the **infidelity** metric from [Captum](https://captum.ai/). A lower infidelity score indicates a more reliable explanation.

```python
infid_score = infidelity(model, perturb_fn, inputs, attributions)
```

Perturbation is done using Gaussian noise. Evaluation is run over batches.

---

## ⚙️ Installation

Make sure you have the following libraries installed:

```bash
pip install torch torchvision captum matplotlib opencv-python
```

---

## 🚀 Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/grape-leaf-disease-explainability.git
cd grape-leaf-disease-explainability
```

### 2. Prepare Dataset

Place the dataset under:

```
/kaggle/input/grape-disease/grape_dataset/train
/kaggle/input/grape-disease/grape_dataset/test
```

### 3. Run Training and Explainability
Modify and run the script provided in `main.py` or your notebook to:
- Load the model
- Load test data
- Generate Grad-CAM++, LRP, combined visualizations
- Compute infidelity scores

---

## 📈 Results

| Technique       | Description                     | Example |
|----------------|----------------------------------|---------|
| Grad-CAM++      | Highlights class-relevant regions | ✅       |
| LRP             | Shows pixel-wise relevance       | ✅       |
| Combined Heatmap| More interpretable explanations  | ✅       |
| Infidelity Score| Quantitative faithfulness        | ✅       |

All visualizations are saved to:
```
outputs/gradcam
outputs/lrp
outputs/combined
```


---

## 🙌 Acknowledgements

- [Captum AI](https://captum.ai/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
