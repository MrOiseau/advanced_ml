# QASPER Dataset Analysis for Master's Thesis

This notebook performs a comprehensive analysis of the **QASPER** dataset, specifically tailored for use in the master's thesis titled:

> "The impact of text chunking methods on information retrieval quality in RAG systems"  
> *(Утицај метода парчања текста на квалитет добављања информација у RAG системима)*

### Notebook Structure

The notebook is structured into the following sections:

### 1. Import Libraries

Imports all necessary Python libraries:

- **pandas** and **numpy** for data manipulation and analysis
- **matplotlib** for data visualization
- **datasets** from Hugging Face to access QASPER dataset
- **collections** and **random** for additional analyses and sampling

### 2. Load Dataset

Loads the complete QASPER dataset (train, validation, and test splits combined) for analysis.

### 3. Basic Overview Analysis

Performs a general statistical overview for each paper in the dataset, including:

- Number of sections per paper
- Total number of paragraphs
- Average paragraph length (in words)
- Total word count per paper
- Number of questions per paper

### 4. Visualization of Basic Metrics

Generates histograms to visualize:
- Number of questions per paper
- Total word count per paper

*(Visualization titles and labels are provided in Cyrillic.)*

### 4. Repeated and Unique Questions Analysis

Analyzes the frequency of question repetitions across papers, identifying:
- Questions repeated across multiple papers
- Questions unique to individual papers

### 5. Detailed QAS Analysis

Performs a detailed analysis of the Question-Answering structure, covering:
- Paragraph statistics (average, maximum, minimum lengths, total words)
- Number of evidence chunks per question (average, max, min)
- Length of free-form answers (average, max, min)

### 5. Export Results to Excel

Exports analytical results into an Excel file (`qasper_combined_overview.xlsx`) containing two sheets:

- **Questions per Paper**: Combines key statistics for easy comparative analysis.
- **Example Q&A**: Contains 10 randomly selected examples of questions, their associated evidence chunks, and generated answers.

### 6. Detailed Visualizations

Generates detailed histograms for:
- Distribution of the number of questions per paper
- Distribution of total words per paper
- Distribution of average paragraph lengths
- Distribution of average free-form answer lengths
- Distribution of average evidence chunks per question

*(All visualizations titles and labels are provided in Cyrillic.)*

---

## Running the Notebook

### Requirements

Ensure that you have installed the required Python packages:

```bash
pip install pandas numpy matplotlib datasets openpyxl
```

### Execution

To run the notebook:

```bash
jupyter notebook
```

Then open and execute cells sequentially.

