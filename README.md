# Text Summarization 


### **Step 1: Setup Virtual Environment & Install Dependencies (5 minutes)**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Alternative: Use the run script**
```bash
chmod +x run.sh
./run.sh
```

**What gets installed:**
- transformers (HuggingFace models)
- torch (PyTorch backend)
- datasets (XSum data)
- streamlit (Web UI)
- pandas (Data handling)

### **Step 2: Run Jupyter Notebook (20 minutes)**

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Install jupyter if not already installed
pip install jupyter

# Run notebook
jupyter notebook demo.ipynb
```


### **Step 3: Run Web App (5 minutes)**

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux

# Run Streamlit app
streamlit run app.py

# OR use the run script
./run.sh
```

**App opens at:** `http://localhost:8501`

**Features:**
- 📄 Summarize your text
- 🎯 Try examples from XSum dataset
- ⚖️ Compare models side-by-side

---

## 📁 Files Included

| File | Purpose |
|------|---------|
| **app.py** | Streamlit web app (localhost) |
| **demo.ipynb** | Jupyter notebook demonstration |
| **requirements.txt** | Dependencies |
| **README.md** | This guide |

---

## 🏆 Model Selection: BART-CNN

### **Why BART-CNN? (Production Choice)**

| Aspect | BART-CNN | PEGASUS-XSum |
|--------|----------|--------------|
| **Output** | 3-4 sentences ✅ | 1 sentence ❌ |
| **Words** | ~45 words | ~12 words |
| **Business Use** | High ✅ | Low ❌ |
| **Industry** | Standard ✅ | Niche ⚠️ |

**Example Output:**

**BART:** "The company reported record earnings of $5B. Cloud services drove growth. CEO announced expansion plans."
(Informative ✅)

**PEGASUS:** "Company reports record earnings."
(Too brief ❌)

---

## 🚀 Usage

### **Jupyter Notebook (Recommended)**

```bash
jupyter notebook demo.ipynb
```

Run all cells to see complete demonstration.

### **Streamlit App**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🔧 Troubleshooting

**Import Error:**
```bash
pip install transformers datasets torch streamlit
```

**CUDA Error:**
```python
# Force CPU
device = -1
```

**Port Busy:**
```bash
streamlit run app.py --server.port 8502
```

---

## 

**Q: Why BART-CNN?**

**A:** Production standard for news summarization. Produces informative 3-4 sentence summaries vs PEGASUS's single sentence. Industry-widely used.

**Q: Libraries?**

**A:** transformers (models), datasets (data), torch (GPU), pandas (processing), streamlit (UI)

**Q: Pipeline?**

**A:** Input → Tokenization → Model → Decoding → Summary (abstracts complexity)

---

## ✅ Before Submitting

- [ ] Notebook runs without errors
- [ ] App accessible at localhost:8501
- [ ] Screenshots captured
- [ ] Understand BART-CNN choice

**Total time: ~30 minutes**

**Good luck! 🎉**
