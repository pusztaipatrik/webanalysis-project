import langid
import os

def remove_punctuation(text):
    # Function to remove punctuation from text
    return ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])

def detect_language(text):
    # Using langid to detect the language of the text
    lang, confidence = langid.classify(text)
    return lang

def collect_matched_words(file_path, keywords):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = remove_punctuation(text)

    language = detect_language(text)

    matched_words = set()
    for keyword in keywords:
        keyword = remove_punctuation(keyword)
        if keyword in text:
            matched_words.add(keyword)

    return matched_words

def save_matched_words_to_file(file_path, matched_words):
    output_file = f"matched_words_{os.path.basename(file_path)}.txt"

    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("\n".join(matched_words))

    print(f"Unique matched words in {os.path.basename(file_path)} saved to {output_file}")

def main():
    # Defining the file paths and keyword file
    text_files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
    keyword_file = "keywords.txt"

    # Reading keywords from the keyword file
    with open(keyword_file, 'r', encoding='utf-8') as keyword_file:
        keywords = keyword_file.read().splitlines()

    # Collecting unique words with at least one match in each text file and saving into file
    for file_path in text_files:
        matched_words = collect_matched_words(file_path, keywords)
        save_matched_words_to_file(file_path, matched_words)

if __name__ == "__main__":
    main()
