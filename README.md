# 🌱 Deep Learning-Based Crop Disease and Insect-Pest Classification Using Convolutional Neural Networks

A deep learning academic project that uses **MobileNetV2 and Convolutional Neural Networks (CNNs)** to classify plant leaf images into crop disease categories. The project includes a trained TensorFlow/Keras model and a Streamlit web application for image-based prediction.

> **Project scope:** The current model is trained on the PlantVillage dataset, containing 38 plant health and disease categories. Insect-pest classification is not yet implemented. The project title describes the broader objective, while the current implementation focuses on crop disease classification.

---

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity. Early identification can help farmers make informed decisions about crop management.

This project uses transfer learning with MobileNetV2 to classify plant leaf images. Users can upload an image through a simple web interface and view the predicted category and confidence score.

### Objectives

* Develop a deep learning model for crop disease classification.
* Apply transfer learning using MobileNetV2.
* Train and evaluate the model using the PlantVillage dataset.
* Build an interactive web application using Streamlit.
* Explore the potential of deep learning in agricultural applications.

---

## ✨ Features

* Upload plant leaf images through a web interface.
* Classify images into 38 trained categories.
* Use MobileNetV2 transfer learning.
* Apply image augmentation during training.
* Display prediction confidence scores.
* Show the top five predicted categories.
* Save the trained model in Keras format.
* Export training history to a CSV file.
* Provide a simple and user-friendly interface.

---

## 🛠️ Technologies Used

| Technology  | Purpose                        |
| ----------- | ------------------------------ |
| Python      | Main programming language      |
| TensorFlow  | Deep learning framework        |
| Keras       | Model development and training |
| MobileNetV2 | Pretrained CNN architecture    |
| NumPy       | Numerical operations           |
| Pillow      | Image processing               |
| Streamlit   | Web application                |
| Git         | Version control                |
| GitHub      | Source code hosting            |
| VS Code     | Development environment        |

---

## 🧠 Model Architecture

The project uses MobileNetV2 with ImageNet pretrained weights as the feature extraction backbone.

The main processing pipeline is:

1. **Image Input:** Accept a plant leaf image.
2. **Preprocessing:** Resize the image to 224 × 224 pixels.
3. **Data Augmentation:** Apply transformations to training images.
4. **Feature Extraction:** Use MobileNetV2 to extract image features.
5. **Global Average Pooling:** Reduce the extracted feature maps.
6. **Dropout:** Help reduce overfitting during training.
7. **Classification:** Use a dense output layer with softmax activation.
8. **Prediction:** Return scores for the 38 trained categories.

The trained model is saved as `crop_classifier.keras`, and the class names are stored in `class_names.json`.

---

## 📂 Project Structure

```text
crop_disease_pest_cnn/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── training_history.csv
│
├── models/
│   ├── crop_classifier.keras
│   └── class_names.json
│
├── training/
│   └── train_model.py
│
├── report/
│   └── REPORT_TEMPLATE.md
│
├── test_dataset.py
└── test_prediction.py
```

**Note:** The virtual environment and dataset are excluded from GitHub to keep the repository manageable.

---

## 📊 Dataset

### PlantVillage Dataset

The project uses the PlantVillage dataset for crop disease classification.

* Approximately 54,000 images are available in the downloaded dataset.
* The dataset contains 38 plant health and disease categories.
* Images are organized into class-specific folders.
* The current model predicts only categories represented in its training data.

**Dataset resources:**

* PlantVillage GitHub repository: https://github.com/spMohanty/PlantVillage-Dataset
* TensorFlow Datasets documentation: https://www.tensorflow.org/datasets/catalog/plant_village

### Future Enhancement: Insect-Pest Classification

Insect-pest classification can be added in a future version using a suitable labeled pest dataset, such as IP102.

* IP102 repository: https://github.com/xpwu95/IP102
* IP102 research paper: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.html

Dataset access and licensing conditions should be reviewed before use.

---

## 💻 System Requirements

* Windows, Linux, or macOS
* Python 3.11 recommended for this project environment
* Git
* Visual Studio Code or another Python IDE
* Internet connection for installing dependencies and downloading pretrained weights when needed
* Sufficient RAM and storage for model training

A GPU is optional. Training can run on a CPU, although it may take longer.

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VIGNESH-04338/crop-disease-pest-cnn.git
```

Navigate into the project directory:

```bash
cd crop-disease-pest-cnn
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Web Application

The repository includes the trained model and class names, so retraining is not required to use the existing application.

```powershell
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually:

`http://localhost:8501`

Upload a plant leaf image to view the model's prediction.

---

## 🏋️ Model Training

To retrain the model, ensure the PlantVillage image folders are available in the directory expected by `training/train_model.py`.

Run:

```powershell
python .\training\train_model.py --epochs 3 --batch-size 16
```

Adjust the number of epochs and batch size according to your available system resources.

The training process generates or updates:

* `models/crop_classifier.keras`
* `models/class_names.json`
* `training_history.csv`

**Note:** Retraining may replace the existing trained model. Keep a backup if you want to preserve the current version.

---

## 📈 Results

The model was trained using MobileNetV2 transfer learning.

| Metric                      |    Result |
| --------------------------- | --------: |
| Number of output categories |        38 |
| Input image size            | 224 × 224 |
| Training epochs             |         3 |
| Final training accuracy     |    93.16% |
| Final validation accuracy   |    93.58% |
| Final validation loss       |    0.1984 |

These results are from the completed training run. Validation accuracy is not the same as independent test accuracy, and it does not guarantee reliable performance on real-world farm images.

---

## ⚠️ Limitations

* The current implementation classifies plant health and disease categories only.
* Insect-pest classification has not yet been implemented.
* The model was trained on PlantVillage images, which may differ from images captured in actual farming environments.
* Lighting, backgrounds, camera quality, and leaf orientation can affect predictions.
* The displayed confidence score is not a guarantee that a prediction is correct.
* An independent test dataset and additional evaluation are needed to assess real-world performance.

This project is an academic prototype and should not replace professional agricultural advice.

---

## 🔮 Future Enhancements

* Integrate a labeled insect-pest dataset.
* Extend the model to recognize additional crop species and diseases.
* Evaluate performance using an independent test dataset.
* Generate a confusion matrix and per-class precision, recall, and F1 scores.
* Improve performance on real-world crop images.
* Add crop-specific disease information and preventive measures.
* Deploy the application to a cloud platform.
* Explore explainable AI techniques to highlight image regions influencing predictions.

---

## 🎓 Academic Purpose

This project demonstrates the application of deep learning, CNN-based image classification, transfer learning, and web application development to an agricultural problem.

It was developed for educational purposes to explore how computer vision can support crop disease identification.

---

## 👨‍💻 Author

**Vignesh Jorige**

B.Tech — Computer Science and Engineering
Specialization: Artificial Intelligence and Machine Learning

GitHub: https://github.com/VIGNESH-04338

---

## 📚 References

1. Mohanty, S. P., Hughes, D. P., and Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science. https://doi.org/10.3389/fpls.2016.01419

2. Wu, X., Zhan, C., Lai, Y.-K., Cheng, M.-M., and Yang, J. (2019). *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

3. PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset

---

**Made with Python, TensorFlow, MobileNetV2, and Streamlit.**
