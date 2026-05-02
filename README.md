# 🤖 SignBridge Pro - ML Edition

**AI-Powered Sign Language Translator with Real ML Backend**

Same beautiful UI, now with real machine learning predictions!

---

## 📁 Project Structure

```
signbridge_pro_ml/
├── app.py                    # Flask backend with ML model
├── download_dataset.py       # Dataset downloader
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── dataset/                # Dataset folder (auto-created)
│   ├── train.csv
│   └── test.csv
│
├── models/                 # Trained models (auto-created)
│   ├── asl_model.keras
│   ├── asl_model_saved/
│   ├── asl_model.tflite
│   ├── best_model.h5
│   ├── label_mapping.json
│   └── training_history.png
│
├── static/                 # Frontend assets
│   ├── style.css          # Same beautiful styling
│   ├── main.js            # Updated with ML integration
│   └── sign-bg.mp4        # Background video
│
└── templates/
    └── index.html         # Same UI
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

```bash
python download_dataset.py
```

This will download the **Sign Language MNIST** dataset or create a synthetic one for testing.

### Step 3: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a CNN model
- Save the model in multiple formats
- Generate training history plots

**Training takes:** 10-30 minutes depending on your hardware

### Step 4: Run the Server

```bash
python app.py
```

Open your browser: **http://localhost:5000**

---

## 🎯 How It Works

### Without Model (Demo Mode)
- Uses local hand tracking (MediaPipe)
- Rule-based letter detection
- Works immediately without training

### With Model (ML Mode)
1. Camera captures hand gesture
2. Frame is sent to Flask backend
3. CNN model processes the image
4. Returns predicted letter + confidence
5. Results displayed in real-time

---

## 📊 Model Architecture

```
Input: 28x28x1 (grayscale hand image)

Conv2D (32 filters) → BatchNorm → ReLU
Conv2D (32 filters) → BatchNorm → ReLU
MaxPool → Dropout(0.25)

Conv2D (64 filters) → BatchNorm → ReLU
Conv2D (64 filters) → BatchNorm → ReLU
MaxPool → Dropout(0.25)

Conv2D (128 filters) → BatchNorm → ReLU
Conv2D (128 filters) → BatchNorm → ReLU
MaxPool → Dropout(0.25)

Flatten
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
Dense(27) → Softmax

Output: 27 classes (A-Z + space)
```

---

## 🔧 Configuration

### Adjust Detection Settings
- **Confidence Threshold**: 0.3 - 0.95
- **Letter Cooldown**: 500ms - 3000ms
- **Frame Buffer Size**: Number of frames to average

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Check model status |
| `/api/predict` | POST | Predict single frame |
| `/api/batch_predict` | POST | Predict multiple frames |
| `/api/save` | POST | Save translation |
| `/api/health` | GET | Health check |

---

## 📈 Improving Accuracy

1. **Better Dataset**: Use real ASL dataset with more samples
2. **Data Augmentation**: Already included (rotation, shift, zoom)
3. **More Epochs**: Increase training epochs
4. **Larger Model**: Add more layers/filters
5. **Hand Cropping**: Crop hand region before prediction

---

## 🐛 Troubleshooting

### "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "Model not found"
Run training first:
```bash
python train_model.py
```

### "Camera not working"
- Allow camera permissions in browser
- Use HTTPS or localhost

### Slow predictions
- Reduce `bufferSize` in main.js
- Use lighter model
- Enable GPU acceleration

---

## 🎨 Same UI Features

✅ Real-time hand tracking  
✅ Beautiful glassmorphism design  
✅ Particle effects & animations  
✅ Text-to-speech  
✅ Sign language guide  
✅ Common sentences  
✅ Text-to-sign converter  
✅ Settings panel  
✅ Toast notifications  

---

## 👨‍💻 Developer

**Y&M** - Software Developer

---

## 📜 License

© 2024 SignBridge Pro. All rights reserved.
