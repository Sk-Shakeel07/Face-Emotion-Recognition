# Face Emotion Recognition

## Overview
This project implements a Face Emotion Recognition system using Python and Machine Learning. The model is trained on facial expression data and can classify emotions in real-time.

## Model Details
- **Face Emotion Recognition Model (62% Accuracy):**
  - Model architecture: `facialemotionmodel.json`
  - Trained weights: `facialemotionmodel.h5`

## Dataset
The model was trained using the **Face Expression Recognition Dataset** available on Kaggle:
[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

## Project Files
- `realtimedetection.py`: Python script for real-time face emotion detection using the trained model.
- `trainmodel.ipynb`: Jupyter Notebook containing the code for training the emotion recognition model.
- `requirements.txt`: List of required Python libraries for running the project.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the appropriate directory.

## Usage
### Train the Model
Run the Jupyter Notebook to train the model:
```bash
jupyter notebook trainmodel.ipynb
```

### Real-time Emotion Detection
Execute the real-time detection script:
```bash
python realtimedetection.py
```

## Requirements
Ensure the following libraries are installed (included in `requirements.txt`):
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Results
- The model achieves **62% accuracy** on the validation dataset.
- It can detect emotions such as happy, sad, angry, surprised, and neutral in real-time.

## Acknowledgments
- Kaggle for providing the dataset.
- Open-source libraries that made this project possible.

## License
This project is open-source and available under the MIT License.

---
Feel free to modify and improve the model for better accuracy!

