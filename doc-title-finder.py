import os
import sys
import shutil
import random
import re
import sqlite3
from pathlib import Path
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from charset_normalizer import from_bytes
from llama_cpp import Llama
import nltk
from nltk.corpus import words


try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

ENGLISH_WORDS = set(w.lower() for w in words.words())


def is_english_word(word):
    cleaned = re.sub(r'^\W+|\W+$', '', word).lower()
    return cleaned in ENGLISH_WORDS

def get_english_factor(sentence):
    tokens = [w for w in sentence.split() if re.sub(r'^\W+|\W+$', '', w)]
    if not tokens:
        return 0.0
    english_words_count = sum(1 for word in tokens if is_english_word(word))
    return english_words_count / len(tokens)

def sanitize_filename(filename):
    """
    Remove problematic characters for Windows/Linux filesystems.
    """
    # Remove smart quotes and regular quotes
    filename = re.sub(r'["“”‘’\']', '', filename)
    # Fix broken hyphenated words (e.g. "Seman- tic" -> "Semantic")
    filename = re.sub(r'(?<=[a-zA-Z])-\s+(?=[a-zA-Z])', '', filename)
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    # Replace newlines and excessive whitespace with a single space
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.strip()
    # Strip trailing periods to prevent "..pdf"
    filename = filename.strip('.')
    return filename

def title_makes_sense(title):
    """
    Basic heuristics to check if the extracted metadata title makes sense.
    """

    print("Checking the title: '{0}'".format(title))

    if not title:
        return False
        
    title = title.strip()
    
    # Titles should be reasonably long
    if len(title) < 10:
        return False
        
    # Titles usually consist of multiple words
    if len(title.split()) < 2:
        return False
        
    title_lower = title.lower()
    bad_words = ["untitled", "microsoft word", "scanned", "document", "unknown", "paper title", "use style", "template", "archive material", "downloaded from"]
    if any(bad in title_lower for bad in bad_words):
        return False
    
    # If title is mostly non-alphanumeric, it's likely gibberish
    alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in title)
    if alphanumeric_chars / max(len(title), 1) < 0.6:
        return False

    print("english factor", get_english_factor(title))

    if get_english_factor(title) < 0.3:
        return False
        
    words = title_lower.split()
    if words:
        # Reject if there is any absurdly long word (LLM token looping)
        max_word_len = max(len(w) for w in words)
        if max_word_len > 35:
            print(f"Title rejected due to extremely long word ({max_word_len} chars).")
            return False

        # Check for LLM repetition loops
        if len(words) >= 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                print(f"Title rejected due to high word repetition (ratio: {len(unique_words) / len(words):.2f}).")
                return False
            
    return True

def infer_title_with_llm(text_content, llm):
    truncated_text = text_content[:4096]
    prompt = f"Document excerpt:\n{truncated_text}\n\nDocument English Title:"
    
    output = llm(
        prompt,
        max_tokens=128,       
        stop=["\n", "\n\n", "Document excerpt:", "Document English Title:"],
        temperature=0.2,     
        top_p=0.9,           
        repeat_penalty=1.1  
    ) 
    raw_text = output['choices'][0]['text'].strip()
    
    # If model wrapped title in quotes (e.g. while thinking), extract it
    quote_match = re.search(r'"([^"\n\r]{5,150})"', raw_text)
    if quote_match and not raw_text.startswith('"'):
        inferred_title = quote_match.group(1).strip()
    else:
        inferred_title = re.sub(
            r'^(the\s+title\s+(of\s+the\s+document\s+)?is[:\s]*|title[:\s]*)',
            '',
            raw_text,
            flags=re.IGNORECASE
        ).strip(' "\'')
    inferred_title = inferred_title.replace('\n', ' ').strip()
    print(f"LLM inferred title: '{inferred_title}'")
    return inferred_title

def main():
    if len(sys.argv) < 4:
        print("Usage: python doc-title-finder.py [light model path] [better model path] [src_folder] [dst_folder]")
        sys.exit(1)

    stupid_llm_model = sys.argv[1]
    better_llm_model = sys.argv[2]
    src_folder = sys.argv[3]
    dst_folder = sys.argv[4]

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    print("#### Initialization ####")
    conn = sqlite3.connect("processed_files.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src_filename TEXT,
        dst_filename TEXT,
        unknown INTEGER,
        rename INTEGER
    )
    """)

    print("Loading LLM...")
    stupid_llm = Llama(
        model_path=stupid_llm_model,
        n_threads=8,
        n_ctx=2048,
        seed=-1,
    )

    better_llm = Llama(
        model_path=better_llm_model,
        n_threads=8,
        n_ctx=2048,
        seed=-1,
    )

    pathlist = list(Path(src_folder).glob('**/*.pdf'))
    pathlist.extend(list(Path(src_folder).glob('**/*.PDF')))
    
    print(f"Found {len(pathlist)} PDF files.")

    for path in pathlist:
        path_str = str(path)
        _, filename = os.path.split(path_str)
        print("\n===========================")
        print(f"Processing: {filename}")

        cursor.execute("SELECT * from processed_files WHERE src_filename = ?", (filename,))
        if cursor.fetchone():
            print(f"omit {filename}")
            continue

        try:
            with open(path_str, 'rb') as fp:
                parser = PDFParser(fp)
                doc = PDFDocument(parser)
                
                # Step 1: Extract metadata title
                title = ""
                if doc.info and len(doc.info) > 0 and "Title" in doc.info[0]:
                    title_b = doc.info[0].get("Title")
                    if isinstance(title_b, bytes):
                        result = from_bytes(title_b).best()
                        if result:
                            decoded_str = str(result)
                            if decoded_str.strip():
                                title = decoded_str.strip()
                    elif isinstance(title_b, str):
                        title = title_b.strip()

                new_title = ""
                if title_makes_sense(title):
                    print(f"Found meaningful metadata title: '{title}'")
                    new_title = title
                else:
                    filename_without_ext = os.path.splitext(filename)[0].replace('_', ' ').strip()
                    if title_makes_sense(filename_without_ext):
                        print(f"Found meaningful title in filename: '{filename_without_ext}'")
                        new_title = filename_without_ext
                    else:
                        print("No meaningful title in metadata or filename. Inferring from content using LLM...")
                        
                        # Step 2: Extract text from the first 2 pages
                        output_string = StringIO()
                        rsrcmgr = PDFResourceManager()
                        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                        interpreter = PDFPageInterpreter(rsrcmgr, device)
                        
                        page_no = 1
                        for page in PDFPage.create_pages(doc):
                            interpreter.process_page(page)	
                            page_no += 1
                            if page_no > 2: # Limit to first 2 pages
                                break
                        
                        text_content = output_string.getvalue().strip()
                        
                        if text_content:
                            inferred_title = infer_title_with_llm(text_content, stupid_llm)
                            
                            if inferred_title and title_makes_sense(inferred_title):
                                new_title = inferred_title
                            else:
                                print("Inferred title rejected by heuristics. Using better LLM")
                                inferred_title = infer_title_with_llm(text_content, better_llm)
                                if inferred_title and title_makes_sense(inferred_title):
                                    new_title = inferred_title
                                else:
                                    print("Inferred title rejected by heuristics. Falling back to filename.")
                                    new_title = filename_without_ext
                        else:
                            print("Warning: Could not extract text from PDF. Falling back to filename.")
                            new_title = filename_without_ext
                        
                # Step 3: Sanitize and prepare final filename
                safe_title = sanitize_filename(new_title)
                if not safe_title:
                    safe_title = f"Document_{random.randint(1000, 9999)}"
                
                # Truncate filename if it's too long
                if len(safe_title) > 150:
                    safe_title = safe_title[:150].strip()
                
                final_filename = safe_title + ".pdf"
                dst_path = os.path.join(dst_folder, final_filename)
                
                # Step 4: Handle naming conflicts
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dst_folder, f"{safe_title[:140]}_{counter}.pdf")
                    counter += 1
                
                print(f"Copying to: {dst_path}")
                shutil.copyfile(path_str, dst_path)

                cursor.execute(
                    "INSERT INTO processed_files (src_filename, dst_filename, unknown, rename) VALUES (?,?,?,?)", 
                    (filename, final_filename, 0, 1 if counter > 1 else 0)
                )
                conn.commit()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    conn.close()
    print("\n#### Finished ####")

if __name__ == "__main__":
    main()
