import os
import json
import nltk
import re
import numpy as np
import torch
import gc
import math
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK models
nltk.download('punkt')

def tokenize_english(text):
    """
    Tokenize English text into sentences using NLTK.
    """
    return nltk.sent_tokenize(text)

def tokenize_hindi(text):
    """
    A simple Hindi tokenizer that splits text by the Hindi danda ("।")
    and newline characters.
    """
    lines = text.split('\n')
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split('।', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part + "।")
    return sentences

def align_many_to_many(english_doc, hindi_doc, max_group_size_en=3, max_group_size_hi=3, threshold=0.6):
    """
    Align the English and Hindi documents using many-to-many mapping via dynamic programming.
    
    - en_sentences: list of English sentences.
    - hi_sentences: list of Hindi sentences.
    - max_group_size_en / hi: maximum consecutive sentences to group on each side.
    - threshold: similarity threshold to flag a grouping as "correct" (1) or not (2).
    
    Returns a list of dictionaries with keys:
      "source": grouped English text,
      "target": grouped Hindi text,
      "similarity": cosine similarity for this grouping,
      "correct": 1 if similarity >= threshold else 2.
    """
    # Tokenize documents
    en_sentences = tokenize_english(english_doc)
    hi_sentences = tokenize_hindi(hindi_doc)
    N = len(en_sentences)
    M = len(hi_sentences)
    
    # Load the LaBSE model once
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    # DP table: dp[i][j] is the best cumulative similarity score aligning first i English and j Hindi sentences.
    dp = [[-math.inf] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0
    # Backpointer table to record which grouping was chosen
    back = [[None] * (M + 1) for _ in range(N + 1)]
    
    # Cache for embeddings to avoid re-computation
    cache = {}
    def get_embedding(text):
        if text in cache:
            return cache[text]
        emb = model.encode(text, convert_to_tensor=True)
        cache[text] = emb
        return emb

    # Fill the DP table by trying all groupings
    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i][j] == -math.inf:
                continue
            # Consider grouping a consecutive block of English sentences
            for a in range(1, max_group_size_en + 1):
                if i + a > N:
                    break
                en_group = " ".join(en_sentences[i:i + a])
                en_emb = get_embedding(en_group)
                # Consider grouping a consecutive block of Hindi sentences
                for b in range(1, max_group_size_hi + 1):
                    if j + b > M:
                        break
                    hi_group = " ".join(hi_sentences[j:j + b])
                    hi_emb = get_embedding(hi_group)
                    # Compute cosine similarity between the grouped texts
                    sim = util.cos_sim(en_emb, hi_emb).item()
                    new_score = dp[i][j] + sim
                    if new_score > dp[i + a][j + b]:
                        dp[i + a][j + b] = new_score
                        back[i + a][j + b] = (a, b, sim)
    
    # Backtrace to recover the segmentation
    segments = []
    i, j = N, M
    while i > 0 or j > 0:
        if back[i][j] is None:
            # Alignment incomplete; break out.
            break
        a, b, sim = back[i][j]
        if a > 1:
            print(f"Error: a={a}, b={b}")
        source_segment = " ".join(en_sentences[i - a:i])
        target_segment = " ".join(hi_sentences[j - b:j])
        correct_flag = 1 if sim >= threshold else 2
        segments.append({
            "source": source_segment,
            "target": target_segment,
            "similarity": sim,
            "correct": correct_flag
        })
        i -= a
        j -= b
    segments.reverse()
    
    # Clean up: release model and clear cache
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return segments

if __name__ == "__main__":
    # Define folder paths
    english_folder = "English"
    hindi_folder = "Hindi"
    output_folder = "Aligned"
    
    os.makedirs(output_folder, exist_ok=True)
    # Process only .txt files in the English folder
    english_files = [f for f in os.listdir(english_folder) if f.endswith('.txt')]
    
    for file_name in english_files:
        english_file = os.path.join(english_folder, file_name)
        hindi_file = os.path.join(hindi_folder, file_name)  # expecting same file name
        
        # Check if corresponding Hindi file exists
        if not os.path.exists(hindi_file):
            print(f"[MISSING] Hindi file not found for: {file_name}")
            continue
        
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_folder, f"aligned_{base_name}.json")
        if os.path.exists(output_file):
            print(f"[SKIP] Already processed: {file_name}")
            continue
        
        try:
            with open(english_file, 'r', encoding='utf-8') as ef:
                english_doc = ef.read()
            with open(hindi_file, 'r', encoding='utf-8') as hf:
                hindi_doc = hf.read()
            
            # print(f"[PROCESSING] Aligning {file_name}")
            aligned = align_many_to_many(english_doc, hindi_doc, 
                                         max_group_size_en=3, 
                                         max_group_size_hi=3, 
                                         threshold=0.6)
            
            with open(output_file, 'w', encoding='utf-8') as outf:
                json.dump(aligned, outf, indent=2, ensure_ascii=False)
            # print(f"[DONE] Processed {file_name}. Output written to {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_name}: {e}")
