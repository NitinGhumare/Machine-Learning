# 🖥️ Process Logger Utility

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A simple **Python utility** to monitor system processes and generate **timestamped logs** at regular intervals.  
Useful for developers, sysadmins, and anyone who wants to keep track of running processes automatically.

---

## ✨ Features
- ✅ Records **PID, Process Name, Username, Memory Usage** (MB).  
- ✅ Creates timestamped **log files** in a specified folder.  
- ✅ Supports both **interactive mode** and **command-line arguments**.  
- ✅ Works on **Linux, macOS, and Windows**.  
- ✅ Lightweight, dependency-free (only `psutil` and `schedule`).  

---

## 📂 Project Structure
```
.
├── process_logger.py     # Main script
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── .gitignore            # Ignore cache, venv, IDE, logs
└── logs/                 # (Optional) Auto-created log folder
```

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/NitinGhumare/process-logger.git
cd process-logger
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Interactive Mode
```bash
python process_logger.py
```
👉 You will be prompted for a folder name and interval (in minutes).

### Command-Line Mode
```bash
python process_logger.py logs 2
```
👉 Creates logs in the `logs/` folder every **2 minutes**.

---

## 📑 Example Output

A sample log file (`process_log_20250930_120000.log`):

```
--------------------------------------------------------------------------------
            System Process Log
    Log created at : Tue Sep 30 12:00:00 2025
--------------------------------------------------------------------------------

PID: 1234     Name: python.exe               User: nitin         Memory: 150.23 MB
PID: 5678     Name: chrome.exe               User: nitin         Memory: 300.78 MB
PID: 9012     Name: explorer.exe             User: nitin         Memory: 120.12 MB

--------------------------------------------------------------------------------
```

---

## 🤝 Contributing
1. Fork the repo 🍴  
2. Create your feature branch (`git checkout -b feature/my-feature`)  
3. Commit changes (`git commit -m "Add my feature"`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a Pull Request 🚀  

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Author
👤 **Nitin G Ghumare**  
📧 [nghumare570@gmail.com](mailto:nghumare570@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/nitin-ghumare) | [GitHub](https://github.com/NitinGhumare)
