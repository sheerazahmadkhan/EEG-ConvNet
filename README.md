# EEG-ConvNet 🧠
A PyTorch-based convolutional neural network pipeline for EEG image classification using k-fold cross-validation, early stopping, and ensemble learning.

---

## 🚀 Features
- Custom CNN architecture built with PyTorch
- Stratified K-Fold cross-validation
- Early stopping & learning rate scheduling
- Confusion matrix visualization with Seaborn
- Ensemble averaging for final evaluation
- Metrics: Accuracy, Precision, Recall, F1-score

---

## 📦 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧰 Dataset
Update the paths in the code or `.env` file:
- **CSV file:** `data/labeledimagesspectrograms100.csv`
- **Image folder:** `data/testdata/`

Each row in the CSV should include:
```
image_name,label
```
where `image_name` points to a file inside `data/testdata/`.

---

## 🧪 Training
Run the model:
```bash
python EEG-ConvNet.py
```

---

## 📊 Outputs
- Training/validation accuracy and loss per epoch
- Confusion matrices for each fold and model
- Final ensemble performance metrics

---

## 🧩 Future Enhancements
- Add TensorBoard logging
- Integrate experiment tracking (e.g., Weights & Biases)
- Save & reload best model weights
- Add hyperparameter tuning

---

## 🧑‍💻 Author
**Sheeraz Ahmad Khan**  
_Deep Learning Researcher | EEG Signal Processing | Computer Vision_


