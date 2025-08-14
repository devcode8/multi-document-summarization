# 📝 Multi-Document Summarization Tool

A powerful, interactive web application that automatically generates summaries from multiple documents using advanced text analysis algorithms. Supports DOCX, TXT, and CSV file formats with step-by-step visualization of the summarization process.

## ✨ Features

- **Multi-Format Support**: Process DOCX, TXT, and CSV files
- **Interactive Visualizations**: Step-by-step analysis with charts and graphs
- **Smart Text Processing**: Advanced preprocessing with stop word and cue word removal
- **Sentence Scoring**: Intelligent scoring based on word frequency and sentence length
- **Customizable Summaries**: Adjustable reduction factor (5% to 50%)
- **Real-time Feedback**: Live processing updates and metrics
- **Export Functionality**: Download generated summaries
- **Multiple Input Methods**: File upload, manual text input, or sample documents

## 🎯 How It Works

The tool uses a sophisticated 4-step process:

1. **Document Preprocessing**: Cleans text, removes stop words, and prepares sentences
2. **Word Frequency Analysis**: Calculates importance scores for each word across all documents
3. **Sentence Scoring**: Scores sentences based on word importance and optimal length
4. **Summary Generation**: Selects top-scoring sentences using threshold and reduction factor

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or Download** this repository to your computer

2. **Open Terminal/Command Prompt** and navigate to the project folder:
   ```bash
   cd path/to/summarization
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter permission issues on macOS/Linux, try:
   ```bash
   pip install --user -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will automatically open in your web browser at `http://localhost:8501`

## 📋 Detailed Installation Guide

### For Windows Users

1. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation: open Command Prompt and type `python --version`

2. **Download the Project**:
   - Download the ZIP file and extract it
   - Or use Git: `git clone [repository-url]`

3. **Install Dependencies**:
   ```cmd
   cd summarization
   pip install streamlit pandas plotly python-docx openpyxl pdfplumber numpy PyPDF2
   ```

4. **Run the Application**:
   ```cmd
   python run.py
   ```

### For macOS Users

1. **Install Python** (if not already installed):
   ```bash
   # Using Homebrew (recommended)
   brew install python

   # Or download from python.org
   ```

2. **Download and Setup**:
   ```bash
   # Navigate to your desired folder
   cd ~/Documents
   
   # If using Git
   git clone [repository-url]
   cd summarization
   
   # Install dependencies
   pip3 install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python3 run.py
   ```

### For Linux Users

1. **Install Python** (usually pre-installed):
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip
   
   # CentOS/RHEL
   sudo yum install python3 python3-pip
   ```

2. **Setup Application**:
   ```bash
   cd ~/Documents
   # Download/clone the project
   cd summarization
   pip3 install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python3 run.py
   ```

## 📖 Usage Guide

### Step 1: Choose Input Method

The application offers three ways to input documents:

#### 🗂️ File Upload
- Click "Upload Files" in the sidebar
- Drag and drop or browse for files
- Supported formats: `.txt`, `.docx`, `.csv`
- Multiple files can be uploaded simultaneously

#### ✍️ Manual Text Input
- Select "Manual Text Input"
- Choose number of documents (1-5)
- Type or paste text directly into the text areas

#### 📋 Sample Documents
- Select "Sample Documents" to use pre-loaded examples
- Perfect for testing the application

### Step 2: Configure Settings

- **Summary Reduction Factor**: Use the sidebar slider (5%-50%)
  - Lower values = shorter summaries
  - Higher values = longer summaries
- **Format Support**: Check which formats are available

### Step 3: Process Documents

1. Click the "🚀 Process Documents" button
2. Watch the interactive visualization unfold:
   - **Preprocessing**: See text cleaning steps
   - **Word Analysis**: View frequency charts
   - **Sentence Scoring**: Analyze scoring with line graphs
   - **Summary Generation**: Watch sentence selection

### Step 4: Download Results

- View the final summary
- See processing statistics
- Click "📥 Download Summary" to save as TXT file

## 🎨 Understanding the Visualizations

### Word Frequency Analysis
- **Bar Charts**: Show most common words across documents
- **Stacked Charts**: Compare word distribution between documents
- **Word Scores Table**: Display normalized importance scores

### Sentence Scoring
- **Line Graphs**: Track primary vs final scores
- **Histograms**: Show score distribution patterns
- **Data Tables**: List top-scoring sentences with details

### Summary Generation
- **Scatter Plots**: Visualize sentence selection process
- **Threshold Lines**: Show selection criteria
- **Selection Status**: Color-coded sentence categories

## ⚠️ Troubleshooting

### Common Issues

**"Streamlit not found" Error**:
```bash
pip install streamlit
```

**"ModuleNotFoundError" for specific packages**:
```bash
pip install [package-name]
```

**Permission denied (macOS/Linux)**:
```bash
pip install --user -r requirements.txt
```

**DOCX files not working**:
```bash
pip install python-docx
```

**Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

### Performance Tips

- **Large Files**: For files > 1MB, consider splitting into smaller documents
- **Multiple Documents**: Process 2-5 documents at once for optimal performance
- **Memory Issues**: Close other applications if processing very large documents

## 📁 Project Structure

```
summarization/
├── README.md                 # This comprehensive guide
├── requirements.txt          # Python dependencies
├── run.py                   # Simple launcher script
├── app.py                   # Main Streamlit application
├── workflow_prototype.py    # Core summarization engine
├── document_processor.py    # Document processing utilities
└── custom_file_input.py    # File input handling
```

## 🛠️ Technical Details

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.25.0 | Web interface framework |
| pandas | ≥1.5.0 | Data manipulation |
| plotly | ≥5.10.0 | Interactive visualizations |
| python-docx | ≥0.8.11 | DOCX file processing |
| numpy | ≥1.21.0 | Numerical operations |
| PyPDF2 | ≥3.0.0 | PDF processing |
| openpyxl | ≥3.0.0 | Excel file support |
| pdfplumber | ≥0.9.0 | Advanced PDF parsing |

### Algorithm Overview

The summarization algorithm uses:
- **TF (Term Frequency)**: Word importance based on occurrence
- **Length Normalization**: Balances sentence length for readability
- **Threshold Selection**: Dynamic cutoff based on average scores
- **Reduction Factor**: User-controlled summary length

## 🤝 Contributing

Found a bug or have a suggestion? Here's how to help:

1. Check if the issue exists in the current version
2. For bugs: provide error messages and steps to reproduce
3. For features: describe the use case and expected behavior

## 📄 License

This project is open-source and available under the MIT License.

## 🆘 Need Help?

### Quick Checklist
- [ ] Python 3.8+ installed?
- [ ] All dependencies installed? (`pip install -r requirements.txt`)
- [ ] Running from correct directory?
- [ ] Port 8501 available?

### Getting Support
- Check the troubleshooting section above
- Verify all dependencies are installed correctly
- Try running with `python -m streamlit run app.py`
- Ensure your Python version is 3.8 or higher

---

**Happy Summarizing! 🎉**

*This tool helps you quickly understand large amounts of text by creating concise, intelligent summaries from multiple documents.*
