# HelmNet — Helmet Detection with CNNs and Transfer Learning

## Project Overview

HelmNet is a computer-vision project that classifies worker images into two safety categories:

- **Without Helmet**
- **With Helmet**

The project compares a custom Convolutional Neural Network (CNN) with transfer-learning approaches based on **VGG16**.

The objective is to build a model that can support automated workplace-safety monitoring by identifying potential helmet-compliance violations from images.

---

## Business Problem

Manual monitoring of personal protective equipment (PPE) compliance can be time-consuming and difficult to scale across large industrial environments.

A computer-vision system can help safety teams:

- detect workers without helmets,
- prioritize risky scenes for review,
- improve monitoring coverage,
- and support preventive safety programs.

Because missing a real safety violation can be costly, the notebook tracks **Violation Recall** in addition to overall classification metrics.

---

## Dataset

The project uses:

```text
images.npy
labels.csv
```

### Image array

```text
Shape: (4125, 200, 200, 3)
Data type: uint8
Pixel range: 0–255
```

Each image is:

```text
200 × 200 pixels
RGB
```

### Labels

```text
0 = Without Helmet
1 = With Helmet
```

Class distribution:

| Class | Count | Approx. Share |
|---|---:|---:|
| Without Helmet | 964 | 23.4% |
| With Helmet | 3161 | 76.6% |

The dataset is therefore moderately imbalanced toward the **With Helmet** class.

> `images.npy` is approximately 495 MB and is intentionally not included in this GitHub package. See `data/README.md` for setup instructions.

---

## Exploratory Data Analysis

The notebook includes:

- dataset shape inspection,
- image datatype and pixel-range checks,
- random-image visualization from each class,
- class-distribution analysis,
- and observations about imbalance.

The random-image inspection is particularly important because it verifies that the numeric labels correspond to the intended helmet/no-helmet classes.

---

## Data Preprocessing

The project uses a stratified split so each subset preserves the original class proportions.

Approximate split:

```text
Training:   2887 images
Validation:  619 images
Test:        619 images
```

Class proportions remain close to:

```text
Without Helmet: 23.4%
With Helmet:    76.6%
```

The notebook also:

- normalizes image pixel values,
- uses class weighting to reduce majority-class bias,
- and keeps the test set untouched until final evaluation.

---

## Models

### 1. CNN from Scratch

A custom CNN is trained directly on the helmet dataset.

The network includes:

- convolutional layers,
- batch normalization,
- max pooling,
- global average pooling,
- dense layers,
- dropout,
- and a sigmoid binary-classification output.

Recorded validation performance:

- Accuracy: **90.15%**
- Precision: **94.60%**
- Macro Recall: **87.58%**
- Violation Recall: **82.76%**
- F1-score: **86.61%**

This model establishes a strong baseline without pretrained visual features.

---

### 2. VGG16 Base

The notebook includes a transfer-learning model using a frozen **VGG16** convolutional base pretrained on ImageNet.

The preserved final notebook contains the architecture and training code so this experiment can be rerun.

---

### 3. VGG16 Base + FFNN

A custom feed-forward classification head is added on top of the frozen VGG16 feature extractor.

The head includes:

- dense layers,
- batch normalization,
- dropout,
- and a sigmoid output layer.

Recorded validation performance:

- Accuracy: **93.70%**
- Precision: **97.80%**
- Macro Recall: **93.49%**
- Violation Recall: **93.10%**
- F1-score: **91.59%**

This was the strongest recorded validation result.

---

### 4. VGG16 + FFNN + Data Augmentation

The transfer-learning model is extended with image augmentation to improve robustness.

Augmentation includes transformations such as:

- horizontal flipping,
- rotation,
- zoom,
- translation,
- and contrast adjustment.

Recorded validation performance:

- Accuracy: **93.05%**
- Precision: **97.16%**
- Macro Recall: **92.35%**
- Violation Recall: **91.03%**
- F1-score: **90.69%**

---

## Model Comparison

Among the recorded validation results:

| Model | Accuracy | Violation Recall | F1-score |
|---|---:|---:|---:|
| CNN from Scratch | 0.9015 | 0.8276 | 0.8661 |
| VGG16 Base + FFNN | **0.9370** | **0.9310** | **0.9159** |
| VGG16 + FFNN + Augmentation | 0.9305 | 0.9103 | 0.9069 |

The notebook selected:

```text
VGG16 Base + FFNN
```

because it achieved the highest recorded validation F1-score while also providing strong violation recall.

---

## Final Test Performance

The selected **VGG16 Base + FFNN** model achieved:

- Accuracy: **92.57%**
- Precision: **97.99%**
- Macro Recall: **92.98%**
- Violation Recall: **93.75%**
- F1-score: **90.23%**

The classification report shows that the model detected approximately **94% of workers without helmets** in the held-out test set.

That makes the model especially useful for safety-monitoring scenarios where missed violations are important.

---

## Key Insights

- Transfer learning substantially improved performance over the custom CNN baseline.
- The VGG16 feature extractor provided useful visual representations despite the relatively small dataset.
- Adding a task-specific FFNN head produced the strongest recorded validation result.
- Data augmentation produced competitive performance but did not outperform the non-augmented FFNN model in the preserved run.
- Accuracy alone is not sufficient because the dataset is imbalanced.
- Violation recall is an important safety-oriented metric because it measures how many actual no-helmet cases are detected.

---

## Business Recommendations

### 1. Use the model as a safety-alerting layer

The system can flag likely helmet violations for human review rather than replacing safety personnel.

### 2. Prioritize violation recall

False negatives represent missed safety violations.

Threshold tuning should therefore consider whether improving violation recall is worth a moderate increase in false alerts.

### 3. Expand the dataset

Collect more images representing:

- different camera angles,
- lighting conditions,
- helmet colors,
- worker poses,
- crowded scenes,
- partial occlusions,
- and multiple workplace environments.

### 4. Monitor production drift

Camera placement, lighting, uniforms, and site conditions can change over time. Monitor performance after deployment and retrain when necessary.

### 5. Consider object detection for real deployments

This project performs image-level binary classification.

A production safety system may benefit from object-detection models that can:

- locate individual workers,
- identify multiple people in one image,
- draw bounding boxes,
- and associate helmet status with each worker.

---

## Repository Structure

```text
helmnet-helmet-detection/
├── README.md
├── HelmNet_Helmet_Detection.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── labels.csv
└── artifacts/
    └── .gitkeep
```

`images.npy` must be provided separately.

---

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Keras
- VGG16
- Transfer Learning
- Data Augmentation
- Google Colab
- Jupyter Notebook

---

# How to Run the Project

## Option 1 — Google Colab

### Step 1: Download or clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/helmnet-helmet-detection.git
cd helmnet-helmet-detection
```

### Step 2: Provide the image dataset

Place your local `images.npy` file at:

```text
data/images.npy
```

The included labels file should already be at:

```text
data/labels.csv
```

### Step 3: Enable GPU

In Colab:

```text
Runtime → Change runtime type → T4 GPU
```

### Step 4: Install dependencies

```python
!pip install -r requirements.txt
```

### Step 5: Open the notebook

```text
HelmNet_Helmet_Detection.ipynb
```

### Step 6: Run all cells

The notebook will:

1. Load the image array and labels
2. Display EDA images
3. Check class imbalance
4. Split the data
5. Normalize images
6. Train the custom CNN
7. Train VGG16 transfer-learning variants
8. Compare validation performance
9. Select the final model
10. Evaluate the final model on the test set
11. Save the selected model and comparison table under `artifacts/`

---

## Option 2 — Run Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/helmnet-helmet-detection.git
cd helmnet-helmet-detection
```

### Step 2: Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Place the dataset

```text
data/images.npy
data/labels.csv
```

### Step 5: Start Jupyter

```bash
jupyter notebook
```

Open:

```text
HelmNet_Helmet_Detection.ipynb
```

and execute the cells in order.

---

## Recommended Environment

```text
Python 3.10+
TensorFlow 2.x
GPU recommended
```

VGG16 weights are downloaded automatically by TensorFlow/Keras when needed.

---

## Output Artifacts

After running the final evaluation section, the notebook saves:

```text
artifacts/helmnet_final_model.keras
artifacts/helmnet_model_comparison.csv
```

These generated files are not required to be committed to GitHub.

---

## Notes

- Deep-learning results can vary slightly across GPU/runtime/library versions.
- The final test set should remain untouched until final model evaluation.
- The project is intended for educational and portfolio purposes.
- Real safety monitoring should include human review and should not rely solely on an automated classifier.

---

## Author

**Sreeman Mandava**

Computer Vision | Machine Learning | AI Engineering | Software Engineering
