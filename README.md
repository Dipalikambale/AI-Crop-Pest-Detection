# AI-Based Crop Pest Detection & Pesticide Recommendation

## 🌱 Project Overview
Farmers often lose crops due to pests or wrong chemical usage.  
This project builds an **end-to-end AI system** to:

- Detect tomato leaf diseases using **CNN (ResNet18 transfer learning)**
- Recommend **safe pesticides** based on disease type
- Provide a **user-friendly web interface** with Streamlit

---

## 🧰 Features
- **Disease Detection**: Classifies tomato leaves as:
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Healthy
- **Rule-based Pesticide Recommendation**: Suggests chemicals based on disease type
- **High Accuracy**: Achieves **~99% validation accuracy**
- **Interactive Web App**: Streamlit interface for easy testing
- **Evaluation Metrics**: Includes precision, recall, F1-score, and confusion matrix

---

## 📂 Project Structure

crop-pest-ai/
├── app/
│ ├── init.py
│ ├── predict.py # Inference and pesticide logic
│ └── app.py # Streamlit UI
├── data/
│ └── raw/ # Tomato leaf images (Early Blight, Late Blight, Leaf Mold, Healthy)
├── model/
│ ├── train.py # Model training script
│ └── pest_model.pth # Trained ResNet18 weights
├── rules/
│ └── pesticide_mapping.json
├── requirements.txt
└── README.md

---

## 🛠 Technologies Used
- **Python**, **PyTorch**, **Torchvision**  
- **Transfer Learning** with **ResNet18**  
- **Streamlit** for web interface  
- **Scikit-learn** for evaluation metrics

---

## ⚡ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/Dipalikambale/AI-Crop-Pest-Detection.git
cd AI-Crop-Pest-Detection/crop-pest-ai
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
streamlit run app/app.py
