# 📄 Enterprise Document Intelligence System

An AI-powered document Q&A system. Upload PDFs and ask questions — get accurate answers with source citations and confidence scoring.

-----

## 📸 Output

![alt text](image.png)

-----

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/enterprise-rag-system.git
cd enterprise-rag-system
```

### 2. Create Virtual Environment

```bash
uv python pin 3.11
uv venv --python 3.11
```

### 3. Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Add API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 6. Run the App

```bash
streamlit run src/app.py
```

Open browser at: http://localhost:8501/


