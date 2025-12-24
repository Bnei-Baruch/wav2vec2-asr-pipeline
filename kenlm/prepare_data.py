import glob
import os
import re
import shutil
import urllib.request
from dotenv import load_dotenv
from lib.db_connection import get_db_connection
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

load_dotenv()

LINKER_URL = os.getenv("LINKER_URL")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "kenlm", "dataset")
ROW_DATA_DIR = os.path.join(PROJECT_ROOT, "kenlm", "row_data")
INPUT_PATTERN = os.path.join(OUTPUT_DIR, "*.txt")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.txt")
RESULT_FILE = os.path.join(OUTPUT_DIR, "train.txt")

LIMIT = 300
query = f"""
    SELECT f.uid FROM files f
    WHERE f.language = 'he'
    AND f.type = 'text'
    AND f.properties->>'insert_type' = 'tamlil'
    limit {LIMIT};
"""

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if os.path.exists(ROW_DATA_DIR):
        shutil.rmtree(ROW_DATA_DIR)
    os.makedirs(ROW_DATA_DIR)

    file_uids = fetch_file_uids()
    for file_uid in file_uids:
        file_uid = file_uid[0]
        print(f"Fetching file text for {file_uid}")
        file_path = fetch_file_text(file_uid)
        if file_path is None:
            continue
        prepare_dataset(file_path, os.path.join(OUTPUT_DIR, f"{file_uid}.txt"))
    split2result()
    print(f"Result file: {RESULT_FILE}")
    print(f"Test file: {TEST_FILE}")

def split2result():
    with open(RESULT_FILE, 'wb') as result_file, open(TEST_FILE, 'wb') as test_file:
        i = 0
        for filename in glob.glob(INPUT_PATTERN):
            if filename == RESULT_FILE or filename == TEST_FILE:
                    continue
            i += 1
            if i % 10 == 0:
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, test_file)
            else:
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, result_file)



def fetch_file_text(file_uid):
    url = f"{LINKER_URL}/{file_uid}"
    print(f"Fetching file text from {url}")
    row_file_path = os.path.join(ROW_DATA_DIR, f"{file_uid}.txt")
    print(f"Saving file text to {row_file_path}")
    try:
        urllib.request.urlretrieve(url, row_file_path)
    except Exception as e:
        print(f"Error fetching file text: {e}")
        return None
    print(f"File text fetched and saved")
    return row_file_path

def fetch_file_uids():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query)
    uids = cur.fetchall()
    return uids


def tokenize_text(text):
    text = re.sub(r'([.,!?;:()\[\]"\'\-])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        content = f_in.read()        
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sent in sentences:
            if not sent.strip():
                continue
            tokenized_sent = tokenize_text(sent)
            f_out.write(tokenized_sent + '\n')

if __name__ == "__main__":
    main()
