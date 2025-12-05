🎭 AI-Powered Sentiment Analyzer (BERT + LoRA)
A full-stack machine learning application that classifies movie reviews into Positive, Negative, or Neutral sentiments.

This project uses a Fine-Tuned BERT model optimized with LoRA (Low-Rank Adaptation) for efficiency, served via a FastAPI backend and visualized with a Streamlit frontend.

Python
FastAPI
Streamlit
HuggingFace

🚀 Features
Three-Class Classification: Accurately detects Positive, Negative, and Neutral sentiments.
Smart "Neutral" Detection: Uses a hybrid approach of Model Confidence + Keyword Analysis to correctly identify average reviews.
Real-Time Analysis: Instant results for single text inputs.
Batch Processing: Analyze multiple reviews at once.
CSV Upload: Upload datasets for bulk analysis.
Interactive Visualizations: Confidence gauges, sentiment distribution pie charts, and histograms.
Downloadable Reports: Export analysis results to CSV.
🛠️ Tech Stack
Backend
Framework: FastAPI
Server: Uvicorn
ML Engine: PyTorch & Hugging Face Transformers
Optimization: PEFT (Parameter-Efficient Fine-Tuning) using LoRA
Frontend
Framework: Streamlit
Visualizations: Plotly Express & Graph Objects
Data Handling: Pandas
Model
Base Model: bert-base-uncased
Fine-Tuning Method: LoRA (Rank=16, Alpha=32)
Training Data: IMDb Movie Reviews (Cleaned)
📂 Project Structure
Bash

movie-review-fine-tuned-model/
│
├── 📂 backend/                  # FastAPI Application
│   ├── 📂 app/
│   │   ├── main.py              # API Endpoints
│   │   ├── model_loader.py      # Model Logic & Smart Filtering
│   │   └── schemas.py           # Pydantic Data Models
│   ├── 📂 model/                # Trained LoRA Adapters (Place files here)
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   └── requirements.txt         # Backend Dependencies
│
├── 📂 frontend/                 # Streamlit Dashboard
│   ├── 📂 .streamlit/
│   │   └── config.toml          # UI Configuration
│   ├── app.py                   # Main Frontend Application
│   └── requirements.txt         # Frontend Dependencies
│
└── README.md                    # Documentation
⚙️ Installation & Setup
Prerequisites
Python 3.10 or 3.11 (Recommended).
Note: Python 3.13 is currently incompatible with PyTorch.
1. Clone/Download the Repository
Navigate to the project folder:

PowerShell

cd movie-review-fine-tuned-model
2. Setup Backend
PowerShell

cd backend
# Optional: Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
3. Setup Frontend
Open a new terminal window:

PowerShell

cd frontend
# Install dependencies
pip install -r requirements.txt
4. Model Setup
Ensure your trained LoRA files are inside backend/model/.
The folder must contain:

adapter_config.json
adapter_model.safetensors
tokenizer_config.json
vocab.txt
special_tokens_map.json
🏃‍♂️ Running the Application
You need to run the Backend and Frontend in two separate terminals.

Terminal 1: Start Backend API
PowerShell

cd backend
python -m uvicorn app.main:app --reload --port 8000
You should see: ✅ Model Loaded Successfully!

Terminal 2: Start Frontend UI
PowerShell

cd frontend
python -m streamlit run app.py
The app will open automatically in your browser at http://localhost:8501.

📖 Usage Guide
1. Single Review
Select "Single Review" from the sidebar.
Type a sentence like "The movie was okay, nothing special."
Click Analyze.
View the Sentiment Label and Confidence Score.
2. Batch Analysis
Select "Batch Analysis".
Paste multiple reviews (one per line).
Click Analyze All.
View summary charts (Pie Chart, Histogram) and detailed table.
3. CSV Analysis
Upload a CSV file containing reviews.
Select the text column.
Analyze and Download the results.
🔌 API Documentation
Once the backend is running, you can access the automatic Swagger documentation:

Docs: http://localhost:8000/docs
Endpoints
GET /: Health check.
POST /predict: Analyze a single string.
POST /predict/batch: Analyze a list of strings.
🧠 Model Logic (Smart Filtering)
To ensure high accuracy even with "safe" models, the backend implements a Hybrid Logic Layer:

Raw Inference: The BERT model predicts Positive/Negative/Neutral.
Keyword Override: If the text contains words like "average", "okay", "decent", "mediocre", the system forces a NEUTRAL label.
Confidence Threshold: If the model's confidence is < 60%, the system defaults to NEUTRAL to avoid false positives/negatives.
📜 License
This project is open-source. Feel free to modify and use it for educational purposes.