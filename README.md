---

# 📌 MinutesAI – Intelligent Meeting Assistant

MinutesAI is a full-stack AI-powered meeting assistant that automatically transcribes, diarises, summarises, and allows users to query their meetings. It integrates state-of-the-art NLP, ML, and AI models to improve productivity, knowledge discovery, and meeting management.

---

## 🚀 Features

### 🎙️ Automatic Transcription

* Converts audio/video into text using **Whisper**.

### 🧍 Speaker Diarisation

* Identifies **who spoke when** using **PyAnnote Audio**.

### 🧾 Abstractive Summarisation

* Generates concise summaries using **BART Large CNN**.

### 🤖 AI Chatbot

* Local LLM (**Phi-3 Mini LoRA fine-tuned**) for question answering.
* Responds with **timestamp references**.
* Uses **SentenceTransformer MiniLM** for semantic retrieval.

### 🔐 Security

* All transcripts, summaries, and diarised text encrypted using **Fernet**.
* Secure login, authentication, and password reset.

### 🌐 Web Platform

* Built with **HTML, CSS, JavaScript, Bootstrap**.
* Users can upload, view, download, and query meeting data.

### 💾 Database

* Stores encrypted data in **MongoDB**.

### 📁 File Upload Limit

* Maximum upload size: **100 MB**

---

## 🛠️ Tech Stack

**Backend:** Flask, Whisper, PyAnnote, BART, SentenceTransformers, Phi-3 Mini LLM, MongoDB
**Frontend:** HTML, CSS, JS, Bootstrap
**Security:** JWT, bcrypt, Fernet Encryption

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MinutesAI.git
cd MinutesAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```
MONGO_URI=your_mongo_uri
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email
MAIL_PASSWORD=your_email_app_password
HUGGINGFACE_TOKEN=your_hf_token
BASE_MODEL_PATH=microsoft/phi-3-mini-4k-instruct
LORA_ADAPTER_PATH=your_lora_adapter_path
MAX_CONTENT_LENGTH=100000000
```

---

## 🔐 Generate Fernet Key

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
with open("fernet.key", "wb") as f:
    f.write(key)
```

---

## ▶️ Run the App

```bash
python app.py
```

Open:
**[http://localhost:5000](http://localhost:5000)**

---

## 🧠 System Workflow

1. Upload meeting file
2. Whisper → Transcription
3. PyAnnote → Speaker diarisation
4. BART → Summarisation
5. Encrypt outputs → Save to MongoDB
6. Embeddings generated for chatbot retrieval
7. Phi-3 Mini → Question answering with timestamps

---

## 📂 Project Structure

```
MinutesAI/
│── app.py
│── requirements.txt
│── fernet.key
│── templates/
│── static/
│── uploads/
└── README.md
```

---

## 🚧 Future Enhancements

* Real-time transcription
* Multi-language support
* Improved diarisation accuracy
* Docker deployment

---

## 🙌 Acknowledgements

* Whisper
* PyAnnote
* BART
* Phi-3 Mini
* HuggingFace
* Flask
* MongoDB

---


