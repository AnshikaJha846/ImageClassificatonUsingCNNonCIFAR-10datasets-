# 🖼️ Image Classification using CNN on CIFAR-10 Dataset
Built a CNN model using PyTorch to classify images from the CIFAR-10 dataset into 10 categories. Achieved ~93% accuracy using Residual Blocks, Batch Normalization &amp; GPU training on Google Colab. 🚀
---
📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 different categories. The model is built using **PyTorch** and trained on a GPU for fast and efficient learning.

Image classification is one of the most fundamental tasks in **Computer Vision** and **Deep Learning**. It has real-world applications in:
- 🚗 Self-driving cars (object detection)
- 🏥 Medical imaging (disease detection)
- 📱 Mobile apps (photo recognition)
- 🔒 Security systems (surveillance)

This project demonstrates how a CNN can learn to recognize objects from raw pixel data with **~93% accuracy**.

---

 📊 Dataset — CIFAR-10

| Property | Details |
|----------|---------|
| Dataset Name | CIFAR-10 |
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 pixels |
| Color Channels | 3 (RGB) |
| Number of Classes | 10 |

🏷️ Classes
| Label | Class |
|-------|-------|
| 0 | ✈️ Airplane |
| 1 | 🚗 Automobile |
| 2 | 🐦 Bird |
| 3 | 🐱 Cat |
| 4 | 🦌 Deer |
| 5 | 🐶 Dog |
| 6 | 🐸 Frog |
| 7 | 🐴 Horse |
| 8 | 🚢 Ship |
| 9 | 🚛 Truck |

---

🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| 🐍 Python 3.8+ | Programming Language |
| 🔥 PyTorch | Deep Learning Framework |
| 📷 Torchvision | Dataset & Image Transforms |
| 🔢 NumPy | Numerical Computations |
| 📊 Matplotlib | Plotting & Visualization |
| 🧪 Scikit-learn | Evaluation Metrics |
| ☁️ Google Colab | GPU Training Environment |

---
 🔄 Project Workflow
Step 1: 📥 Data Loading
- CIFAR-10 dataset loaded using `torchvision.datasets.CIFAR10`
- Split into training (50,000) and testing (10,000) images
- DataLoader used for batch processing

Step 2: ⚙️ Data Preprocessing
- Images normalized with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`
- Converted to PyTorch tensors
- Data augmentation applied (random flips, crops)
- Batched into groups of 128 images

Step 3: 🏗️ Model Building
- Custom CNN architecture with Residual Blocks
- 11,155,274 trainable parameters
- Batch Normalization for stable training
- Dropout for regularization

Step 4: 🏋️ Training
- Optimizer: Adam with learning rate scheduling
- Loss Function: Cross Entropy Loss
- Trained for 50 epochs on GPU (CUDA)
- Best model checkpoint saved automatically

Step 5: 📈 Evaluation
- Accuracy and Loss tracked per epoch
- Confusion Matrix generated
- Per-class accuracy calculated

Step 6: 🔮 Prediction
- Load any image
- Preprocess and resize to 32×32
- Model predicts class with confidence score

---

🧠 Model Architecture

```
Input Image (3 × 32 × 32)
        ↓
┌─────────────────────┐
│  Conv Layer 1       │  → 64 filters, 3×3, ReLU
│  Batch Norm         │
│  Max Pooling 2×2    │
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Residual Block 1   │  → 128 filters, 3×3, ReLU
│  Batch Norm         │
│  Skip Connection    │
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Residual Block 2   │  → 256 filters, 3×3, ReLU
│  Batch Norm         │
│  Skip Connection    │
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Residual Block 3   │  → 512 filters, 3×3, ReLU
│  Batch Norm         │
│  Skip Connection    │
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Global Avg Pooling │
│  Dropout (0.5)      │
│  Fully Connected    │  → 512 → 10
│  Softmax Output     │  → 10 classes
└─────────────────────┘
        ↓
   Prediction 🎯
```

Key Components:
- **Convolutional Layers** — Extract spatial features from images
- **ReLU Activation** — Introduces non-linearity
- **Batch Normalization** — Stabilizes and speeds up training
- **Residual Connections** — Helps train deeper networks
- **Max Pooling** — Reduces spatial dimensions
- **Dropout** — Prevents overfitting
- **Softmax Output** — Converts scores to class probabilities

---

⚙️ Installation & Setup

1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/cifar10-cnn.git
cd cifar10-cnn
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Train the Model
```bash
python train.py
```

4. Evaluate the Model
```bash
python evaluate.py
```

5. Predict on a New Image
```bash
python predict.py --image path/to/your/image.jpg
```

6. Run on Google Colab (Recommended)
- Open [Google Colab](https://colab.research.google.com)
- Enable GPU: Runtime → Change Runtime Type → T4 GPU
- Upload all project files and run `train.py`

---

 📈 Results

### Training Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99% |
| Validation Accuracy | **~93%** |
| Training Loss | 0.58 |
| Validation Loss | 0.65 |
| Total Epochs | 50 |
| Training Time (GPU) | ~1 hour |

Per-Class Accuracy

| Class | Accuracy |
|-------|---------|
| ✈️ Airplane | 95% |
| 🚗 Automobile | 97% |
| 🐦 Bird | 92% |
| 🐱 Cat | 88% |
| 🦌 Deer | 94% |
| 🐶 Dog | 91% |
| 🐸 Frog | 97% |
| 🐴 Horse | 96% |
| 🚢 Ship | 97% |
| 🚛 Truck | 96% |
---
🚀 Future Improvements

- 📊 **Data Augmentation** — Add rotation, color jitter for better generalization
- 🔧 **Hyperparameter Tuning** — Experiment with learning rates and batch sizes
- 🏗️ **Advanced Architectures** — Try ResNet-50, EfficientNet, or Vision Transformers
- 📱 **Model Deployment** — Deploy as a web app using Flask or FastAPI
- 🗜️ **Model Compression** — Use quantization for faster inference on mobile
- 📦 **Larger Datasets** — Train on CIFAR-100 or ImageNet
- 🔄 **Transfer Learning** — Use pretrained models for better accuracy

---
🏁 Conclusion

This project successfully demonstrates how a **CNN with Residual Blocks** can achieve **~93% accuracy** on the CIFAR-10 dataset. Key takeaways:

- ✅ CNNs are powerful for image classification tasks
- ✅ Residual connections help train deeper networks effectively
- ✅ GPU training reduces time from days to under 1 hour
- ✅ Batch Normalization and Dropout prevent overfitting
- ✅ Model generalizes well across all 10 classes

This project serves as a strong foundation for more advanced computer vision work.

---
[results.zip](https://github.com/user-attachments/files/27322555/results.zip)
[files (1).zip](https://github.com/user-attachments/files/27322542/files.1.zip)
