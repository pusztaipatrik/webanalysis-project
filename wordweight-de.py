from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def get_top_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Manually define a list of German stop words
    german_stop_words = [
        "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", "anderem",
        "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf", "aus", "bei", "bin",
        "bis", "bist", "da", "damit", "dann", "der", "den", "des", "dem", "die", "das", "dass", "daß", "derselbe",
        "derselben", "denselben", "desselben", "demselben", "dieselbe", "dieselben", "dasselbe", "dazu", "dein", "deine",
        "deinem", "deinen", "deiner", "deines", "denn", "derer", "dessen", "dich", "dir", "du", "dies", "diese", "diesem",
        "diesen", "dieser", "dieses", "doch", "dort", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "einig",
        "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "er", "ihn", "ihm", "es", "etwas", "euer", "eure",
        "eurem", "euren", "eurer", "eures", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten",
        "hier", "hin", "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "euch", "im",
        "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener",
        "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte", "machen",
        "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner",
        "meines", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr",
        "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich", "sie", "ihnen", "sind", "so", "solche",
        "solchem", "solchen", "solcher", "solches", "soll", "sollte", "sondern", "sonst", "über", "um", "und", "uns",
        "unse", "unsem", "unsen", "unser", "unses", "unter", "viel", "vom", "von", "vor", "während", "war", "waren",
        "warst", "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde",
        "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu",
        "zum", "zur", "zwar", "zwischen"
    ]

    vectorizer = TfidfVectorizer(stop_words=german_stop_words)
    tfidf_matrix = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Combine feature names and corresponding TF-IDF scores into a dictionary
    words_and_weights = dict(zip(feature_names, tfidf_scores))

    # Sort the dictionary by TF-IDF scores in descending order
    sorted_words_and_weights = sorted(words_and_weights.items(), key=lambda x: x[1], reverse=True)

    return sorted_words_and_weights

def create_wordcloud(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(12, 6))
    
    # Display the word cloud
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Display the top words and their weights
    plt.subplot(1, 2, 2)
    top_words = get_top_words(file_path)
    words, weights = zip(*top_words[:10])  # Display top 10 words
    plt.barh(words, weights, color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.title('Top Words and Their TF-IDF Scores')

    plt.tight_layout()
    plt.show()

def main():
    # Define the file paths
    text_files = ["file1.txt", "file2.txt"]

    # Analyze each text file and create word cloud with top words
    for file_path in text_files:
        create_wordcloud(file_path)

if __name__ == "__main__":
    main()
