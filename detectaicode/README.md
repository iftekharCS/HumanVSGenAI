# 🧪 DetectAICode

Project to detect AI code using simple perturbation strategy using datasets stored in the `data/` directory.

## 📁 Project Structure

```
.
├── data/ # Dataset folder
│ └── ...
├── src/ # Core analysis and detection logic
│ └── ...
├── main.py # Entry point to run the project
├── analyze_lexical.py # Lexical analysis script
├── analyze_logrank.py # Log-rank analysis script
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Getting Started

### 1. Clone the Repository

```bash
git clone detectaicode
cd detectaicode
```

### 2. Set Up a Virtual Environment

On Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Analysis Scripts

```bash
python analyze_lexical.py
python analyze_logrank.py
```

### 5. Running the Project

```bash
python main.py
```

## Configuration Notes

This project uses tree-sitter for code parsing functionality. Depending on your environment, you may need additional configuration or installation of language grammars.

For more details, see:
[Tree-Sitter-Github](https://github.com/Goldziher/tree-sitter-language-pack)

Compatibility
Tested on:
Ubuntu 22.04
Windows 10

The project should work on any OS with Python 3.7+ installed. Some OS-specific configuration may be needed for tree-sitter.
