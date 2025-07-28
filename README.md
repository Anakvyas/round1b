# Adobe Challenge 1B: Persona-Driven Document Intelligence

## Overview

This solution builds upon the Round 1A PDF processing foundation to implement a persona-driven document intelligence system. It extracts and prioritizes relevant sections from PDF documents based on a specific persona and job-to-be-done, optimized for speed and efficiency.

---

## Approach

### 1. Enhanced PDF Processing (Building on Round 1A)
- **Reuses Round 1A Architecture**: Leverages the existing block extraction, table detection, and footer removal logic  
- **Optimized Text Extraction**: Uses PyMuPDF for fast, reliable text extraction  
- **Smart Section Detection**: Identifies document sections based on headings, font sizes, and formatting patterns  
- **Table Awareness**: Filters out table content from section analysis using Round 1A table detection

### 2. Lightweight Persona Analysis
- **Keyword-Based Matching**: Uses pre-defined persona and job type mappings for fast analysis  
- **Domain-Specific Keywords**: Maps personas (researcher, student, analyst, etc.) to relevant terms  
- **Job Type Detection**: Identifies job categories (review, analysis, research, learning) for targeted extraction  
- **No Heavy NLP**: Avoids large models to meet size and speed constraints

### 3. Fast Section Ranking
- **Keyword Scoring**: Uses simple but effective keyword frequency and position scoring  
- **Multi-Factor Ranking**: Combines title relevance, content matches, and structural importance  
- **Heading Level Weighting**: Prioritizes higher-level headings (H1 > H2 > H3)  
- **Length Normalization**: Balances short vs. long sections appropriately

### 4. Intelligent Subsection Extraction
- **Paragraph Segmentation**: Breaks sections into meaningful paragraphs  
- **Content Filtering**: Selects substantial paragraphs (>80 characters)  
- **Hierarchical Preservation**: Maintains relationship between sections and subsections  

---

## Key Optimizations for Challenge Constraints

### Speed Optimizations (≤60 seconds)
- No Heavy ML Models
- Efficient PDF Processing (reuses Round 1A logic)
- Smart Caching
- Minimal Dependencies (PyMuPDF, pdfplumber, Pillow)

### Memory Optimizations (≤1GB)
- Streaming Processing
- Minimal Data Structures
- Garbage Collection

### Size Constraints
- No Large Models
- Minimal Dependencies
- Efficient Code

---

## Libraries Used

### Core Libraries (Minimal Set)
- `PyMuPDF (fitz)`: Fast PDF text extraction and processing
- `pdfplumber`: Table extraction and structured content analysis
- `Pillow`: Image processing support for complex PDFs

---

## Key Design Decisions

- No `scikit-learn`: Reduced size and complexity  
- No `NLTK`: Used regex and string ops  
- No `numpy`: Native data structures only  

---

## Architecture

![alt text](image.png)


## Build and Run Instructions

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t adobe-1b-solution:latest .
```
### Running the Solution

```
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-1b-solution:latest

```

## Input Format
Place the following in the input directory:

config.json Example

```
{
  "persona": "PhD Researcher in Computational Biology with expertise in machine learning",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies and performance benchmarks",
  "documents": ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
}

```


## Output Format
Generates challenge1b_output.json with structure like:

```
{
  "metadata": {
    "input_documents": ["paper1.pdf", "paper2.pdf"],
    "persona": "PhD Researcher...",
    "job_to_be_done": "Prepare a comprehensive...",
    "processing_timestamp": "2025-07-28T..."
  },
  "extracted_sections": [
    {
      "document": "paper1.pdf",
      "page_number": 11,
      "section_title": "Methodology",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "paper1.pdf",
      "subsection_id": "paper1.pdf_section_1_para_1",
      "refined_text": "The proposed methodology combines...",
      "page_number": 11
    }
  ]
}

```

## Key Innovations

- **Multi-Modal Scoring**: Combines content similarity, structural importance, and keyword relevance  
- **Persona-Specific Focus**: Adapts ranking based on persona type and domain  
- **Hierarchical Processing**: Maintains document structure while extracting relevant content  
- **Robust PDF Handling**: Handles various PDF formats and structures  
- **Scalable Architecture**: Processes multiple documents efficiently  

---

## Performance Characteristics

- **Execution Time**: <60 seconds for 3–5 documents  
- **Memory Usage**: <1GB  
- **Model Size**: No large models  
- **Hardware**: CPU-only, optimized for `amd64`  

---

## Error Handling

- Graceful PDF processing failures  
- Fallback scoring mechanisms  
- Input validation and error logging  
- Robust extraction with multiple libraries  

---


### Document Types Supported

- Academic research papers  
- Business/financial reports  
- Educational content  
- Technical documentation  

### Persona Types Supported

- Researchers and academics  
- Students and learners  
- Business analysts  
- Technical professionals  
