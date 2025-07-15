# NLP and Naive Bayes Classifier from Scratch
import re
import numpy as np
import pandas as pd

# --- 1. Text Preprocessing ---

# Using NLTK's stemmer and stopwords is standard practice, as building these
# from scratch is a separate, complex project. Here, we build the *pipeline* from scratch.
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text, stemmer, custom_stopwords):
    """
    Cleans and tokenizes a single text document.

    Args:
        text (str): The raw text string.
        stemmer: An instance of a stemmer (e.g., PorterStemmer).
        custom_stopwords (set): A set of stopwords to remove.

    Returns:
        str: A cleaned string of space-separated tokens.
    """
    # Keep only alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase and split into words (tokenize)
    tokens = text.lower().split()
    # Remove stopwords and apply stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in custom_stopwords]
    # Join tokens back into a single string
    return " ".join(stemmed_tokens)


# --- 2. CountVectorizer (Bag of Words) from Scratch ---

class MyCountVectorizer:
    """
    Converts a collection of text documents to a matrix of token counts.
    """
    def __init__(self, max_features=1500):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.stop_words_ = None

    def fit_transform(self, corpus):
        """
        Learns the vocabulary dictionary and returns the document-term matrix.

        Args:
            corpus (list of str): A list of text documents.

        Returns:
            np.ndarray: The document-term matrix (Bag of Words).
        """
        word_counts = {}
        # Count frequencies of all words in the corpus
        for document in corpus:
            for word in document.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort words by frequency and select the top `max_features`
        sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        top_words = [word for word, count in sorted_words[:self.max_features]]

        # Create the vocabulary mapping word to index
        self.vocabulary_ = {word: i for i, word in enumerate(top_words)}

        # Create the document-term matrix
        doc_term_matrix = np.zeros((len(corpus), len(self.vocabulary_)), dtype=int)
        for i, document in enumerate(corpus):
            for word in document.split():
                if word in self.vocabulary_:
                    doc_term_matrix[i, self.vocabulary_[word]] += 1

        return doc_term_matrix


# --- 3. Train-Test Split from Scratch ---

def my_train_test_split(X, y, test_size=0.20, random_state=None):
    """
    Splits arrays or matrices into random train and test subsets.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target labels.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    if random_state:
        np.random.seed(random_state)

    # Shuffle indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Determine split point
    split_idx = int(X.shape[0] * (1 - test_size))

    # Split indices into train and test
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create data splits
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


# --- 4. Multinomial Naive Bayes from Scratch ---

class MyMultinomialNB:
    """
    A Multinomial Naive Bayes classifier for text data.
    """
    def __init__(self, alpha=1.0):
        # alpha is the Laplace smoothing parameter
        self.alpha = alpha
        self._priors = {}
        self._likelihoods = {}
        self._classes = []

    def fit(self, X, y):
        """
        Trains the Naive Bayes classifier.

        Args:
            X (np.ndarray): The training feature vectors.
            y (np.ndarray): The training target labels.
        """
        n_docs, n_features = X.shape
        self._classes = np.unique(y)

        for c in self._classes:
            # Get all documents belonging to the current class
            X_c = X[y == c]

            # Calculate prior probability P(c)
            self._priors[c] = len(X_c) / n_docs

            # Calculate likelihood P(word | c) with Laplace smoothing
            # Sum word counts for all docs in class c, then add alpha
            word_counts_in_class = X_c.sum(axis=0) + self.alpha
            # Total word count in class c, adjusted for smoothing
            total_words_in_class = word_counts_in_class.sum()
            
            # We store log likelihoods to prevent underflow and speed up prediction
            self._likelihoods[c] = np.log(word_counts_in_class / total_words_in_class)

    def predict(self, X):
        """
        Predicts the class labels for a given set of feature vectors.

        Args:
            X (np.ndarray): The feature vectors to predict.

        Returns:
            np.ndarray: The predicted class labels.
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """Helper to predict one document."""
        posteriors = {}
        for c in self._classes:
            # Start with log prior P(c)
            log_prior_c = np.log(self._priors[c])
            # Add sum of log likelihoods P(word | c) weighted by word counts in x
            log_likelihood_c = (self._likelihoods[c] * x).sum()
            # The posterior is proportional to prior * likelihood
            posteriors[c] = log_prior_c + log_likelihood_c

        # Return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get)

# --- 5. Evaluation Metrics from Scratch ---

def accuracy_score(y_true, y_pred):
    """Calculates classification accuracy."""
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    """Computes the confusion matrix."""
    classes = np.unique(y_true)
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        true_label_idx = np.where(classes == y_true[i])[0][0]
        pred_label_idx = np.where(classes == y_pred[i])[0][0]
        matrix[true_label_idx, pred_label_idx] += 1
    return matrix


# --- 6. Main Execution Block ---
if __name__ == '__main__':
    dataset = pd.DataFrame(data)
    print("--- 1. Original Data ---")
    print(dataset)

    # Initialize tools for preprocessing
    stemmer = PorterStemmer()
    # Handle the 'not' exception by removing it from the standard stopword list
    custom_stopwords = set(stopwords.words('english'))
    custom_stopwords.remove('not')

    # Apply preprocessing to the corpus
    corpus = [preprocess_text(review, stemmer, custom_stopwords) for review in dataset['Review']]
    
    print("\n--- 2. Cleaned and Stemmed Corpus ---")
    print(corpus)

    # Create Bag of Words representation
    vectorizer = MyCountVectorizer(max_features=1500) # Using 100 features for this small dataset
    X = vectorizer.fit_transform(corpus)
    y = dataset.iloc[:, -1].values
    

    # Split data into training and testing sets
    #X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.25, random_state=0)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Train the Naive Bayes classifier
    classifier = MyMultinomialNB(alpha=1.0)
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- 4. Model Evaluation ---")
    print(f"Test Set Predictions: {y_pred}")
    print(f"Actual Test Set Labels: {y_test}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
