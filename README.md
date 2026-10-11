# Deep Learning-Based Crop Disease and Insect-Pest Classification Using Convolutional Neural Networks

A Python + TensorFlow/Keras + Streamlit academic project that classifies crop/leaf images into the categories represented in the training dataset. It uses a CNN architecture based on MobileNetV2 transfer learning, image augmentation, validation split, early stopping, and a Streamlit upload interface.

> **Important scope note:** PlantVillage contains crop health/disease classes, but not insect-pest classes. To honestly cover both parts of the project title, add pest-category images (for example, from IP102) to the same class-folder dataset before training. The model can only predict classes it was trained on.

## Features

- Image upload interface built with Streamlit
- CNN-based image classification using MobileNetV2
- Data augmentation: random flip, rotation, and zoom
- 80/20 training-validation split
- Model checkpointing and early stopping
- Top-5 predictions and confidence scores
- Class names saved alongside the trained model
- Training history exported to CSV

## Tech stack

- Python
- TensorFlow / Keras
- CNN (MobileNetV2 transfer learning)
- Streamlit
- NumPy and Pillow
- Visual Studio Code
- Git and GitHub

## Project structure

```text
crop_disease_pest_cnn/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── images/
│       └── <one-folder-per-class>/
├── models/
│   └── (trained model files created after training)
└── training/
    └── train_model.py
```

## 1. Requirements

Install Python 3.11 (recommended for a smooth TensorFlow setup), Git, and VS Code. In VS Code, open this project folder and select the Python interpreter from the virtual environment.

Check your Python version in the VS Code terminal:

```powershell
python --version
```

## 2. Create a virtual environment (Windows PowerShell)

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 3. Prepare the dataset

The training code expects one directory per class, with images inside each class directory. Example:

```text
dataset/
└── images/
    ├── Apple___Apple_scab/
    │   ├── image001.jpg
    │   └── image002.jpg
    ├── Tomato___healthy/
    │   ├── image001.jpg
    │   └── image002.jpg
    ├── Tomato___Early_blight/
    │   └── image001.jpg
    ├── Pest___Aphids/
    │   └── image001.jpg
    └── Pest___Armyworm/
        └── image001.jpg
```

Use real class names that match your dataset. Do not create fake or empty classes.

### Crop disease dataset: PlantVillage

PlantVillage contains roughly 54,000 labeled images across 38 crop health/disease classes. Options:
- Dataset project/repository: https://github.com/spMohanty/PlantVillage-Dataset
- TensorFlow Datasets description and dataset citation: https://www.tensorflow.org/datasets/catalog/plant_village
- Mendeley Data listing: https://data.mendeley.com/datasets/tywbtsjrjv/1

Download/extract the image class folders and copy the class folders into `dataset/images/`. Avoid copying duplicate augmented images into both train and validation data if you create your own split manually; this starter uses a random 80/20 split.

### Insect-pest dataset: IP102

IP102 is a research dataset with more than 75,000 images across 102 insect-pest categories. Review its official repository and usage terms:
- Official repository and download details: https://github.com/xpwu95/IP102
- CVPR paper page: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.html

IP102 may be provided with split files and a different directory layout. Arrange/copy its images into class folders under `dataset/images/` before training. Keep the original class names where possible. Check the dataset's academic-use conditions before using it outside academic research.

**Important:** If you train only with PlantVillage, the project will classify plant health/disease classes only, not insect pests. For the full title, combine suitable disease and pest class folders. This starter treats each folder as one output class; it does not separately predict “disease vs pest” as a second output.

## 4. Train the model

From the project root, with the virtual environment activated:

```powershell
python training/train_model.py --epochs 10
```

For a quick test with a small sample dataset:

```powershell
python training/train_model.py --epochs 2 --batch-size 8
```

The first run may download MobileNetV2 pretrained weights and requires internet access. Training time depends on your CPU/GPU, image count, and memory. If the computer runs out of memory, reduce `--batch-size` to `8` or `16`.

When training finishes, the project creates:
- `models/crop_classifier.keras`
- `models/class_names.json`
- `training_history.csv`

The model and dataset are excluded from Git by `.gitignore` because they can be large. Do not upload them unless your instructor requires it and you have checked GitHub file-size limits and dataset licensing.

## 5. Run the web application

After successful training:

```powershell
streamlit run app.py
```

Streamlit will show a local address in the terminal (usually `http://localhost:8501`). Open it in your browser, upload a JPG/PNG image, and review the top predictions.

## 6. How the model works

1. **Input:** Load a crop/leaf/pest image.
2. **Resize:** Convert it to 224 × 224 pixels.
3. **Augmentation:** Randomly flip, rotate, and zoom training images.
4. **CNN feature extraction:** MobileNetV2 convolution layers learn visual features such as edges, textures, shapes, and patterns.
5. **Pooling:** Global average pooling compresses the feature maps.
6. **Dropout:** Reduces overfitting during training.
7. **Classification:** A softmax layer returns a score for each training class.
8. **Output:** The interface displays the highest-scoring class and top-five scores.

## 7. Evaluation and limitations

- The script reports validation loss and validation accuracy.
- Validation accuracy is not proof of field performance. Images from the same source or near-duplicates can inflate scores.
- PlantVillage images often have clean backgrounds, while real farm images have varied lighting, backgrounds, and leaf angles.
- For a stronger evaluation, create a separate test set that is never used during training or model selection. Report accuracy, per-class precision/recall/F1, and a confusion matrix.
- A confidence score is not a guarantee. Do not treat the app as a substitute for an agricultural expert or use it alone to choose pesticides.
- Because this is a multi-class classifier, it selects one class. Images containing multiple pests/diseases or no relevant subject may be misclassified.

## 8. GitHub upload

First create a **new empty repository** on GitHub (do not initialize it with a README if you already have this README locally). Then run from the project root:

```powershell
git init
git add .
git status
git commit -m "Initial crop disease and pest CNN project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL. Before `git add .`, confirm that your dataset and trained model are not staged:

```powershell
git status
```

If Git prompts for authentication, follow GitHub's current sign-in flow. Never put passwords or access tokens into source code or commit them to the repository.

## 9. Suggested report chapters

1. Abstract
2. Introduction
3. Problem statement and objectives
4. Literature review
5. Dataset description and licensing
6. Proposed methodology
7. CNN / MobileNetV2 architecture
8. Implementation and system requirements
9. Results: training curves, validation metrics, confusion matrix, sample predictions
10. Limitations and future scope
11. Conclusion
12. References

Only report results you actually obtained by training and testing the model. Do not invent accuracy values or claim the model detects pest categories unless those categories were included in training.

## References

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science. https://doi.org/10.3389/fpls.2016.01419
- Wu, X., Zhan, C., Lai, Y.-K., Cheng, M.-M., & Yang, J. (2019). *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition*. CVPR, 8787–8796.
