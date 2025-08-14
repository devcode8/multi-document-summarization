"""
Multi-Document Extractive Summarization Workflow Prototype
Based on Key Phrase Extraction methodology from the research paper
"""

import re
from collections import Counter
from typing import List, Dict, Tuple

class MultiDocumentSummarizer:
    def __init__(self):
        self.stop_words = {
            'i', 'my', 'me', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        self.cue_words = {
            'there', 'here', 'so', 'also', 'however', 'therefore', 'thus',
            'hence', 'moreover', 'furthermore', 'additionally', 'meanwhile',
            'consequently', 'accordingly', 'nevertheless', 'nonetheless'
        }
        
        self.basic_dictionary_words = {
            'get', 'got', 'getting', 'come', 'came', 'coming', 'go', 'going',
            'went', 'gone', 'make', 'made', 'making', 'take', 'took', 'taken',
            'taking', 'give', 'gave', 'given', 'giving', 'say', 'said', 'saying',
            'see', 'saw', 'seen', 'seeing', 'know', 'knew', 'known', 'knowing',
            'think', 'thought', 'thinking', 'look', 'looked', 'looking'
        }

    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Step 1: Preprocess documents by removing stop words, cue words, and basic dictionary words
        """
        processed_docs = []
        
        for doc in documents:
            # Convert to lowercase and split into sentences
            sentences = self._split_into_sentences(doc.lower())
            processed_sentences = []
            
            for sentence in sentences:
                # Remove punctuation and split into words
                words = re.findall(r'\b[a-z]+\b', sentence)
                
                # Remove stop words, cue words, and basic dictionary words
                filtered_words = [
                    word for word in words 
                    if word not in self.stop_words 
                    and word not in self.cue_words 
                    and word not in self.basic_dictionary_words
                ]
                
                if filtered_words:  # Only keep non-empty sentences
                    processed_sentences.append(' '.join(filtered_words))
            
            processed_docs.append(processed_sentences)
        
        return processed_docs

    def calculate_word_scores(self, processed_docs: List[List[str]]) -> Dict[str, float]:
        """
        Step 2: Calculate word scores based on frequency and importance
        """
        word_freq = Counter()
        
        # Count word frequencies across all documents
        for doc in processed_docs:
            for sentence in doc:
                words = sentence.split()
                word_freq.update(words)
        
        # Calculate scores (higher frequency = higher score)
        max_freq = max(word_freq.values()) if word_freq else 1
        word_scores = {
            word: freq / max_freq 
            for word, freq in word_freq.items()
        }
        
        return word_scores

    def calculate_sentence_scores(self, processed_docs: List[List[str]], 
                                word_scores: Dict[str, float]) -> List[List[Tuple[str, float, float]]]:
        """
        Step 3: Calculate sentence scores using primary and final scoring
        """
        scored_docs = []
        all_sentence_lengths = []
        
        # First pass: calculate primary scores and collect lengths
        for doc in processed_docs:
            doc_scores = []
            for sentence in doc:
                words = sentence.split()
                # Primary score = sum of word scores
                primary_score = sum(word_scores.get(word, 0) for word in words)
                sentence_length = len(words)
                all_sentence_lengths.append(sentence_length)
                doc_scores.append((sentence, primary_score, sentence_length))
            scored_docs.append(doc_scores)
        
        # Calculate average length
        avg_length = sum(all_sentence_lengths) / len(all_sentence_lengths) if all_sentence_lengths else 1
        
        # Second pass: calculate final scores
        final_scored_docs = []
        for doc_scores in scored_docs:
            final_doc_scores = []
            for sentence, primary_score, length in doc_scores:
                # Final score formula: FS = Pi * (AL / Si)
                final_score = primary_score * (avg_length / length) if length > 0 else 0
                final_doc_scores.append((sentence, primary_score, final_score))
            final_scored_docs.append(final_doc_scores)
        
        return final_scored_docs

    def generate_summaries(self, scored_docs: List[List[Tuple[str, float, float]]], 
                          reduction_factor: float = 0.1) -> List[str]:
        """
        Step 4: Generate document summaries using threshold-based selection
        """
        doc_summaries = []
        
        for doc_scores in scored_docs:
            if not doc_scores:
                doc_summaries.append("")
                continue
                
            # Calculate threshold as average of final scores
            final_scores = [score[2] for score in doc_scores]
            threshold = sum(final_scores) / len(final_scores)
            
            # Select sentences above threshold
            selected_sentences = [
                (sentence, final_score) 
                for sentence, _, final_score in doc_scores 
                if final_score >= threshold
            ]
            
            # Sort by score descending
            selected_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Apply reduction factor (limit number of sentences)
            max_sentences = max(1, int(len(doc_scores) * reduction_factor))
            selected_sentences = selected_sentences[:max_sentences]
            
            # Create summary
            summary = '. '.join([sent[0] for sent in selected_sentences])
            doc_summaries.append(summary)
        
        return doc_summaries

    def reduce_summary(self, summaries: List[str], max_sentences_per_doc: int = 4) -> str:
        """
        Step 5: Final summary reduction by removing subset sentences and limiting length
        """
        all_sentences = []
        
        # Collect all sentences from all summaries
        for summary in summaries:
            if summary:
                sentences = [s.strip() for s in summary.split('.') if s.strip()]
                all_sentences.extend(sentences)
        
        # Remove subset sentences
        filtered_sentences = []
        for i, sent1 in enumerate(all_sentences):
            is_subset = False
            for j, sent2 in enumerate(all_sentences):
                if i != j and sent1 in sent2 and len(sent1) < len(sent2):
                    is_subset = True
                    break
            if not is_subset:
                filtered_sentences.append(sent1)
        
        # Limit to maximum sentences per document ratio
        max_total_sentences = len(summaries) * max_sentences_per_doc
        if len(filtered_sentences) > max_total_sentences:
            filtered_sentences = filtered_sentences[:max_total_sentences]
        
        return '. '.join(filtered_sentences) + '.'

    def summarize(self, documents: List[str], reduction_factor: float = 0.1) -> str:
        """
        Main workflow: Complete multi-document summarization process
        """
        # Step 1: Preprocess documents
        processed_docs = self.preprocess_documents(documents)
        
        # Step 2: Calculate word scores
        word_scores = self.calculate_word_scores(processed_docs)
        
        # Step 3: Calculate sentence scores
        scored_docs = self.calculate_sentence_scores(processed_docs, word_scores)
        
        # Step 4: Generate individual document summaries
        doc_summaries = self.generate_summaries(scored_docs, reduction_factor)
        
        # Step 5: Reduce and combine summaries
        final_summary = self.reduce_summary(doc_summaries)
        
        return final_summary

    def _split_into_sentences(self, text: str) -> List[str]:
        """Helper method to split text into sentences"""
        sentences = re.split(r'[.;:]', text)
        return [s.strip() for s in sentences if s.strip()]

