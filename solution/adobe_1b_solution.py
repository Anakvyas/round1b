#!/usr/bin/env python3
"""
Adobe Challenge 1B: Persona-Driven Document Intelligence
Optimized solution building on Round 1A code structure
"""

import os
import json
import sys
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import math

# Reuse Round 1A utilities
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io

class PDFProcessor:
    """Enhanced PDF processor building on Round 1A approach"""
    
    def __init__(self):
        self.table_cache = {}
    
    def extract_blocks_from_pdf(self, pdf_path):
        """Enhanced version of Round 1A block extraction"""
        doc = fitz.open(pdf_path)
        blocks = []
        page_heights = {}

        for page_num, page in enumerate(doc, 1):
            page_heights[page_num] = page.rect.height
            words = page.get_text("dict")

            if not words.get("blocks"):
                continue

            for block in words["blocks"]:
                if "lines" not in block:
                    continue
                
                text = ""
                spans = []
                is_bold = False
                font_sizes = []
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            spans.append(span)
                            text += span["text"] + " "
                            font_name = span.get("font", "").lower()
                            flags = span.get("flags", 0)
                            if "bold" in font_name or (flags & 2):
                                is_bold = True
                            font_sizes.append(span.get("size", 12))

                if not spans or not text.strip():
                    continue

                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                bbox = block.get("bbox") or spans[0].get("bbox")
                block_text = text.strip().replace("\n", " ")

                blocks.append({
                    "text": block_text,
                    "font_size": avg_font_size,
                    "bbox": bbox,
                    "page": page_num,
                    "is_bold": is_bold,
                    "is_centered": self._is_centered(page, bbox)
                })

        doc.close()
        return blocks, page_heights
    
    def _is_centered(self, page, bbox):
        """Check if text block is centered"""
        page_width = page.rect.width
        block_x0, _, block_x1, _ = bbox
        block_center = (block_x0 + block_x1) / 2
        return abs(block_center - page_width / 2) < 50
    
    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables using pdfplumber"""
        if pdf_path in self.table_cache:
            return self.table_cache[pdf_path]
        
        all_tables = []
        table_texts = set()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:
                            continue
                        
                        cleaned_table = [[cell.strip() if cell else "" for cell in row] for row in table]
                        
                        # Simple table validation
                        if len(cleaned_table) < 2 or len(cleaned_table[0]) < 2:
                            continue
                        
                        flat_text = " ".join(cell for row in cleaned_table for cell in row if cell)
                        table_texts.add(flat_text.strip().lower())
                        
                        all_tables.append({
                            "page": page_num,
                            "table": cleaned_table
                        })
        except Exception as e:
            print(f"Warning: Table extraction failed for {pdf_path}: {e}")
        
        result = (all_tables, table_texts)
        self.table_cache[pdf_path] = result
        return result
    
    def process_document(self, pdf_path):
        """Process single PDF document"""
        try:
            blocks, page_heights = self.extract_blocks_from_pdf(pdf_path)
            tables, table_texts = self.extract_tables_from_pdf(pdf_path)
            
            # Remove footers (simplified version)
            clean_blocks = self._remove_footers(blocks, page_heights)
            
            # Extract sections
            sections = self._extract_sections(clean_blocks, table_texts)
            
            return {
                'filename': os.path.basename(pdf_path),
                'sections': sections,
                'total_pages': max(page_heights.keys()) if page_heights else 0
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {
                'filename': os.path.basename(pdf_path),
                'sections': [],
                'total_pages': 0
            }
    
    def _remove_footers(self, blocks, page_heights):
        """Simplified footer removal"""
        footer_candidates = defaultdict(list)
        threshold = 70  # pixels from bottom
        
        for block in blocks:
            y0 = block["bbox"][1]
            page = block["page"]
            page_height = page_heights.get(page, 800)
            
            if y0 >= page_height - threshold:
                text = re.sub(r'\d+', '', block["text"]).strip().lower()
                if text:
                    footer_candidates[text].append(page)
        
        # Find repeated footers
        total_pages = max(page_heights.keys()) if page_heights else 1
        repeated_footers = set()
        
        for text, pages in footer_candidates.items():
            if len(set(pages)) / total_pages >= 0.5:  # Appears on 50%+ of pages
                repeated_footers.add(text)
        
        # Remove footer blocks
        clean_blocks = []
        for block in blocks:
            norm_text = re.sub(r'\d+', '', block["text"]).strip().lower()
            if norm_text not in repeated_footers:
                clean_blocks.append(block)
        
        return clean_blocks
    
    def _extract_sections(self, blocks, table_texts):
        """Extract document sections"""
        # Identify headings
        headings = []
        font_sizes = [b["font_size"] for b in blocks if b["font_size"] > 8]
        
        if not font_sizes:
            return []
        
        avg_font = sum(font_sizes) / len(font_sizes)
        
        for block in blocks:
            if self._is_likely_heading(block, avg_font, table_texts):
                headings.append(block)
        
        # Sort headings by page and position
        headings.sort(key=lambda h: (h["page"], h["bbox"][1]))
        
        # Create sections
        sections = []
        current_section = None
        
        for i, block in enumerate(blocks):
            if block in headings:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': block["text"].strip(),
                    'level': self._get_heading_level(block, headings),
                    'page': block["page"],
                    'text': '',
                    'blocks': []
                }
            elif current_section:
                # Add content to current section
                current_section['text'] += ' ' + block["text"]
                current_section['blocks'].append(block)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # If no headings found, create page-based sections
        if not sections:
            page_content = defaultdict(list)
            for block in blocks:
                page_content[block["page"]].append(block)
            
            for page_num, page_blocks in page_content.items():
                text = ' '.join(b["text"] for b in page_blocks)
                sections.append({
                    'title': f'Page {page_num} Content',
                    'level': 'H3',
                    'page': page_num,
                    'text': text,
                    'blocks': page_blocks
                })
        
        return sections
    
    def _is_likely_heading(self, block, avg_font, table_texts, threshold=0.5):
        """Determine if block is likely a heading"""
        text = block["text"].strip()
        font_size = block["font_size"]
        word_count = len(text.split())
        
        # Basic filters
        if len(text) < 3 or word_count > 20:
            return False
        
        # Check if it's table content
        text_lower = text.lower()
        for table_text in table_texts:
            if text_lower in table_text or table_text in text_lower:
                return False
        
        # Date patterns
        if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
            return False
        
        # Heading indicators
        if block["is_bold"] or block["is_centered"]:
            return True
        
        if font_size > avg_font * 1.1 and word_count <= 10:
            return True
        
        # Numbered sections
        if re.match(r'^\d+(\.\d+)*\.?\s+\w+', text):
            return True
        
        return False
    
    def _get_heading_level(self, heading, all_headings):
        """Determine heading level based on font size and structure"""
        font_sizes = sorted(set(h["font_size"] for h in all_headings), reverse=True)
        
        # Check for numbered structure first
        text = heading["text"].strip()
        if re.match(r'^\d+\.\d+\.\d+', text):
            return "H3"
        elif re.match(r'^\d+\.\d+', text):
            return "H2"
        elif re.match(r'^\d+\.', text):
            return "H1"
        
        # Use font size hierarchy
        font_size = heading["font_size"]
        if len(font_sizes) >= 1 and font_size >= font_sizes[0]:
            return "H1"
        elif len(font_sizes) >= 2 and font_size >= font_sizes[1]:
            return "H2"
        else:
            return "H3"

class PersonaAnalyzer:
    """Lightweight persona analysis using keyword matching"""
    
    def __init__(self):
        # Pre-defined persona mappings for efficiency
        self.persona_keywords = {
            'researcher': ['research', 'study', 'analysis', 'methodology', 'data', 'results', 'findings', 'literature', 'academic', 'scientific'],
            'student': ['learn', 'understand', 'concept', 'theory', 'example', 'practice', 'exam', 'study', 'definition', 'explanation'],
            'analyst': ['trend', 'performance', 'metric', 'data', 'insight', 'report', 'analysis', 'business', 'financial', 'market'],
            'engineer': ['technical', 'implementation', 'system', 'design', 'specification', 'architecture', 'development', 'solution'],
            'manager': ['strategy', 'overview', 'summary', 'decision', 'management', 'planning', 'objective', 'goal', 'outcome']
        }
        
        self.job_keywords = {
            'review': ['summary', 'overview', 'comparison', 'evaluation', 'assessment'],
            'analysis': ['data', 'trends', 'patterns', 'insights', 'metrics', 'performance'],
            'research': ['methodology', 'findings', 'results', 'literature', 'studies', 'evidence'],
            'learning': ['concepts', 'principles', 'theory', 'examples', 'practice', 'understanding']
        }
    
    def analyze_requirements(self, persona, job_to_be_done):
        """Extract key requirements from persona and job"""
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Identify persona type
        persona_type = 'general'
        for p_type, keywords in self.persona_keywords.items():
            if any(keyword in persona_lower for keyword in keywords):
                persona_type = p_type
                break
        
        # Identify job type
        job_type = 'general'
        for j_type, keywords in self.job_keywords.items():
            if any(keyword in job_lower for keyword in keywords):
                job_type = j_type
                break
        
        # Extract key terms
        key_terms = self._extract_key_terms(persona + ' ' + job_to_be_done)
        
        # Get relevant keywords
        relevant_keywords = set()
        relevant_keywords.update(self.persona_keywords.get(persona_type, []))
        relevant_keywords.update(self.job_keywords.get(job_type, []))
        relevant_keywords.update(key_terms[:10])  # Top 10 extracted terms
        
        return {
            'persona_type': persona_type,
            'job_type': job_type,
            'key_terms': list(relevant_keywords),
            'priority_keywords': key_terms[:5]
        }
    
    def _extract_key_terms(self, text):
        """Extract key terms from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common words
        common_words = {'with', 'that', 'this', 'from', 'they', 'were', 'been', 'have', 'their', 'said', 'each', 'which', 'what', 'there', 'will', 'would', 'could', 'should', 'about', 'other', 'many', 'some', 'very', 'when', 'much', 'also', 'well', 'more', 'only', 'just', 'than', 'over', 'such', 'even', 'most', 'made', 'after', 'first', 'being', 'through', 'where', 'before', 'between', 'both', 'under', 'within', 'without'}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Count frequency and return top terms
        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(20)]

class SectionRanker:
    """Fast section ranking using keyword matching and simple scoring"""
    
    def __init__(self):
        pass
    
    def rank_sections(self, documents, requirements):
        """Rank sections by relevance"""
        all_sections = []
        
        # Collect all sections
        for doc in documents:
            for section in doc['sections']:
                section_info = {
                    'document': doc['filename'],
                    'title': section['title'],
                    'text': section['text'],
                    'level': section['level'],
                    'page': section['page'],
                    'importance_rank': 0,
                    'score': 0
                }
                all_sections.append(section_info)
        
        if not all_sections:
            return []
        
        # Calculate scores
        key_terms = requirements['key_terms']
        priority_keywords = requirements['priority_keywords']
        
        for section in all_sections:
            score = self._calculate_section_score(section, key_terms, priority_keywords)
            section['score'] = score
        
        # Sort by score
        all_sections.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, section in enumerate(all_sections):
            section['importance_rank'] = i + 1
        
        return all_sections
    
    def _calculate_section_score(self, section, key_terms, priority_keywords):
        """Calculate section relevance score"""
        title = section['title'].lower()
        text = section['text'].lower()
        combined = title + ' ' + text
        
        score = 0
        
        # Title keyword matches (higher weight)
        for term in key_terms:
            if term.lower() in title:
                score += 2
        
        # Priority keyword matches
        for keyword in priority_keywords:
            count = combined.count(keyword.lower())
            score += count * 1.5
        
        # General keyword matches
        for term in key_terms:
            count = combined.count(term.lower())
            score += count * 0.5
        
        # Heading level bonus
        level_bonus = {'H1': 1.0, 'H2': 0.7, 'H3': 0.5}.get(section['level'], 0.3)
        score += level_bonus
        
        # Length normalization (prevent very short sections from dominating)
        text_length = len(section['text'])
        if text_length < 100:
            score *= 0.7
        elif text_length > 1000:
            score *= 1.2
        
        return score

class SubsectionExtractor:
    """Extract meaningful subsections from ranked sections"""
    
    def extract_subsections(self, ranked_sections, max_subsections=20):
        """Extract subsections from top sections"""
        subsections = []
        
        for section in ranked_sections[:10]:  # Top 10 sections
            paragraphs = self._split_into_paragraphs(section['text'])
            
            for i, paragraph in enumerate(paragraphs[:3]):  # Max 3 paragraphs per section
                if len(paragraph.strip()) > 80:  # Minimum paragraph length
                    subsection = {
                        'document': section['document'],
                        'section_title': section['title'],
                        'subsection_id': f"{section['document']}_section_{section['importance_rank']}_para_{i+1}",
                        'refined_text': paragraph.strip(),
                        'page_number': section['page']
                    }
                    subsections.append(subsection)
        
        return subsections[:max_subsections]
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs"""
        # Split by double newlines or sentence boundaries
        paragraphs = re.split(r'\n\s*\n|(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter and clean paragraphs
        clean_paragraphs = []
        for para in paragraphs:
            clean_para = para.strip()
            if len(clean_para) > 50 and not re.match(r'^\d+$', clean_para):
                clean_paragraphs.append(clean_para)
        
        return clean_paragraphs

class Challenge1BProcessor:
    """Main processor optimized for speed and efficiency"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.persona_analyzer = PersonaAnalyzer()
        self.section_ranker = SectionRanker()
        self.subsection_extractor = SubsectionExtractor()
    
    def process(self, input_dir, output_dir):
        """Main processing function"""
        start_time = time.time()
        
        # Read configuration
        config_path = os.path.join(input_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Error: config.json not found in {input_dir}")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        persona = config.get('persona', '')
        job_to_be_done = config.get('job_to_be_done', '')
        document_files = config.get('documents', [])
        
        print(f"Processing {len(document_files)} documents...")
        
        # Process documents
        documents = []
        for doc_file in document_files:
            doc_path = os.path.join(input_dir, doc_file)
            if os.path.exists(doc_path):
                print(f"Processing: {doc_file}")
                doc_info = self.pdf_processor.process_document(doc_path)
                documents.append(doc_info)
            else:
                print(f"Warning: {doc_file} not found")
        
        if not documents:
            print("No documents processed successfully")
            return
        
        # Analyze requirements
        print("Analyzing persona and job requirements...")
        requirements = self.persona_analyzer.analyze_requirements(persona, job_to_be_done)
        
        # Rank sections
        print("Ranking sections...")
        ranked_sections = self.section_ranker.rank_sections(documents, requirements)
        
        # Extract subsections
        print("Extracting subsections...")
        subsections = self.subsection_extractor.extract_subsections(ranked_sections)
        
        # Generate output
        output = {
            "metadata": {
                "input_documents": document_files,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section['document'],
                    "page_number": section['page'],
                    "section_title": section['title'],
                    "importance_rank": section['importance_rank']
                }
                for section in ranked_sections[:15]  # Top 15 sections
            ],
            "subsection_analysis": [
                {
                    "document": subsection['document'],
                    "subsection_id": subsection['subsection_id'],
                    "refined_text": subsection['refined_text'],
                    "page_number": subsection['page_number']
                }
                for subsection in subsections
            ]
        }
        
        # Save output
        output_path = os.path.join(output_dir, 'challenge1b_output.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        print(f"Extracted {len(ranked_sections)} sections and {len(subsections)} subsections")

def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process
    processor = Challenge1BProcessor()
    processor.process(input_dir, output_dir)

if __name__ == "__main__":
    main()
