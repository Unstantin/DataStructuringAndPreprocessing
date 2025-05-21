import csv
import math
import random
from collections import defaultdict


class NaiveBayesSpamClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()

    def clean_text(self, text):
        text = text.lower()
        for ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
            text = text.replace(ch, ' ')
        return text.split()

    def train(self, texts, labels):
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_words_per_class = defaultdict(int)

        for text, label in zip(texts, labels):
            class_counts[label] += 1
            words = self.clean_text(text)
            for word in words:
                word_counts[label][word] += 1
                total_words_per_class[label] += 1
                self.vocab.add(word)

        total_docs = sum(class_counts.values())
        for label in class_counts:
            self.class_probs[label] = class_counts[label] / total_docs

        vocab_size = len(self.vocab)
        for label in class_counts:
            self.word_probs[label] = {}
            denominator = total_words_per_class[label] + self.alpha * vocab_size
            for word in self.vocab:
                count = word_counts[label].get(word, 0)
                self.word_probs[label][word] = (count + self.alpha) / denominator

    def predict(self, text):
        words = self.clean_text(text)
        max_log_prob = -float('inf')
        best_class = None

        for label in self.class_probs:
            log_prob = math.log(self.class_probs[label])
            for word in words:
                if word in self.word_probs[label]:
                    log_prob += math.log(self.word_probs[label][word])

            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = label

        return best_class


def load_dataset(filename, test_ratio=0.2, encoding='windows-1251'):
    with open(filename, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader)
        data = [(row[0], row[1]) for row in reader if len(row) >= 2]

    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    train = data[:split_idx]
    test = data[split_idx:]

    X_train = [text for (label, text) in train]
    y_train = [label for (label, text) in train]
    X_test = [text for (label, text) in test]
    y_test = [label for (label, text) in test]

    return X_train, y_train, X_test, y_test


def evaluate_model(classifier, X_test, y_test):
    correct = 0
    for text, true_label in zip(X_test, y_test):
        pred = classifier.predict(text)
        if pred == true_label:
            correct += 1
    return correct / len(X_test)


if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = load_dataset('spamdb.csv', encoding='windows-1251')
    except FileNotFoundError:
        print("Ошибка: файл 'sms_spam.csv' не найден!")
        exit()

    nb_classifier = NaiveBayesSpamClassifier(alpha=1.0)
    nb_classifier.train(X_train, y_train)

    accuracy = evaluate_model(nb_classifier, X_test, y_test)
    print(f"Точность модели: {accuracy:.1%}")