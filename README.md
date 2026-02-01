arkdown
# Structural Jewelry Search Engine

A visual search tool designed for finding structurally similar gold rings from a large catalog. This project uses **DINOv2** (Meta's self-supervised vision transformer) to analyze ring geometry and shapes, allowing users to find matches even if the lighting, rotation, or background differs.

## 🚀 Features
* **Visual Search:** Upload an image of a ring to find the closest matches in the database.
* **AI-Powered:** Uses DINOv2 for deep feature extraction (identifying shapes, not just colors).
* **Vector Search:** Fast similarity matching using cosine similarity.
* **Smart Detection:** Integrates YOLOv8 to automatically detect and crop rings from uploaded photos for better accuracy.

## 📂 Project Structure
* `app.py`: The main web application (built with Streamlit).
* `search.py`: Core logic for AI feature extraction and image comparison.
* `reset_db.py`: Script to scan the `catalog/` folder and rebuild the search index.
* `setup_models.py`: Automates the downloading of necessary AI models (YOLO, DINOv2).
* `catalog/`: Directory containing the reference images (Gold Rings).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ayesha859/jewelary_search_engine.git](https://github.com/ayesha859/jewelary_search_engine.git)
   cd jewelary_search_engine
Install dependencies:

Bash
pip install -r requirements.txt
Initialize the models and database: First, download the AI models:

Bash
python setup_models.py
Then, index the current images in the catalog:

Bash
python reset_db.py
Run the Application:

Bash
streamlit run app.py
📋 Requirements
Python 3.10+

PyTorch / Torchvision

Streamlit

Pillow

OpenCV


---
