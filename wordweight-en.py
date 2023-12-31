from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def get_top_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    words_and_weights = dict(zip(feature_names, tfidf_scores))

    sorted_words_and_weights = sorted(words_and_weights.items(), key=lambda x: x[1], reverse=True)

    return sorted_words_and_weights

def create_wordcloud(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(12, 6))
    
    # Displaying the word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Displaying the top words and their weights
    plt.subplot(1, 2, 2)
    top_words = get_top_words(file_path)
    words, weights = zip(*top_words[:10])  # Displaying the top 10 words
    plt.barh(words, weights, color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.title('Top Words and Their TF-IDF Scores')

    plt.tight_layout()
    plt.show()

def main():
    # Defining the file paths
    text_files = ["file1.txt", "file2.txt"]

    # Analyzing each text file and creating a word cloud with the top words
    for file_path in text_files:
        create_wordcloud(file_path)

if __name__ == "__main__":
    main()
