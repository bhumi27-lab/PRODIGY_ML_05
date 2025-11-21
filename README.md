# 🍽️ Task-05: Food Image Classification & Calorie Prediction

This project uses a Convolutional Neural Network (CNN) to automatically **recognize food items from images** and then **estimate their calorie, protein, fat, and carbohydrate content**.  
It is built using the **Food-11 dataset**, trained in Google Colab, and deployed with a simple prediction function.

---

## 📦 Clone This Repository

To clone the project, run:

```bash
git clone https://github.com/bhumi27-lab/PRODIGY_ML_05.git
cd PRODIGY_ML_05
```



---

## 🚀 Features

- 🔍 **Food Recognition (11 Categories)**
- 🔥 **Calorie + Nutrition Prediction**
- 📷 Upload any food image (dataset or real-world)
- 🧠 CNN model trained from scratch
- 📈 Evaluation on validation & test sets
- 💾 Saved model (`.h5`) + metadata (`.json`)

---

## 📂 Dataset Information

**Food-11 Dataset (11 classes)**  
Downloaded from:  
https://www.kaggle.com/datasets/trolukovich/food11-image-dataset

Folder structure:

```
training/
│── Bread
│── Dairy product
│── Dessert
│── Egg
│── Fried food
│── Meat
│── Noodles-Pasta
│── Rice
│── Seafood
│── Soup
│── Vegetable-Fruit
```

---

## 🧠 Model Architecture (CNN)

- Conv2D → BatchNorm → MaxPool  
- Conv2D → BatchNorm → MaxPool  
- Conv2D → BatchNorm → MaxPool  
- Flatten → Dense(256) → Dropout(0.5) → Softmax  

Trained for ~20 epochs with EarlyStopping & ModelCheckpoint.

---

## 🧪 Evaluation

The model was evaluated on the **evaluation** split of Food-11 using:

- Accuracy  
- Loss  
- Classification Report (precision, recall, f1-score)

Typical accuracy: **60–75%** depending on GPU.

---

## 🔮 Prediction + Nutrition System

After training, the system can:

1. Take an uploaded food image  
2. Predict the food class  
3. Return nutrition values such as:  
   - Calories  
   - Protein  
   - Fat  
   - Carbs  

Example output:

```
Predicted Food: Noodles-Pasta
Nutrition Info: {'calories': 138, 'protein': 4.5, 'fat': 2.1, 'carbs': 25}
```

---

## 📦 Files Included in This Repository

```

class_labels.json                  → mapping of class index to food name
nutrition_data.json                → calorie & nutrient values
Task5_food_classification.ipynb    → full Google Colab notebook
requirements.txt                   → list of dependencies
README.md                          → project documentation
```

---

## ▶️ How to Use

### 1️⃣ Load the model  
```python
from tensorflow.keras.models import load_model
model = load_model("food_classification_model.h5")
```

### 2️⃣ Load label & nutrition maps  
```python
import json

class_labels = json.load(open("class_labels.json"))
nutrition = json.load(open("nutrition_data.json"))
```

### 3️⃣ Predict food item  
```python
food, info = predict_food("your_image.jpg")
print(food, info)
```

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Google Colab  
- Kaggle Dataset  

---

## ⚠️ License

This project has **NO LICENSE**.  
It is created **strictly for educational and academic purposes only**.  
You may view the code but **not use it commercially**.

---

## 💡 Future Improvements

- Transfer Learning (MobileNet, EfficientNet)  
- Real-time calorie estimation using portion size  
- Gradio / Streamlit UI  
- API deployment using FastAPI  

---

## 👩‍💻 Author

**BHUMI SIRVI**  
Machine Learning Intern — Prodigy InfoTech
