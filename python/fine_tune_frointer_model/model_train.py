import argparse
import os
import math
import json
import random
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from items import Item
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotly.graph_objects as go
from collections import Counter
from testing import Tester
# Used to reduce high dimensions to 3D
from sklearn.manifold import TSNE
from tqdm import tqdm
# Imports for traditional machine learning
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# NLP related imports
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import simple_preprocess
# Finally, more imports for more advanced machine learning
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

# Create model classes set
MODEL_CLASSES = {"random", "constant", "linear_regression", "bag_of_words", "word2vec"}

TOP_ELECTRONICS_BRANDS = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]

# Predict a random price
class RamdonPriceModel:
    def __init__(self):
        random.seed(42)

    # This function will be called for each item by Tester.run_datapoint(i)
    def _random_pricer(self, _):
        """
        We need to pass in data because the Tester.test requires the data parameter
        """
        return random.randrange(1,1000)

    def run(self, data):
        Tester.test(self._random_pricer, data)

# Always predict the average training data price 
class ConstantPriceModel:
    def __init__(self, train):
        prices = [item.price for item in train]
        self._avg_train_price = sum(prices) / len(prices)

    # This function will be called for each item by Tester.run_datapoint(i)
    def _constant_pricer(self, _):
        return self._avg_train_price

    def run(self, data):
        Tester.test(self._constant_pricer, data)

# Linear regression model
class LinearRegressionModel:
    def __init__(self, train, test):
        np.random.seed(42)
        self._model = LinearRegression()
        self._train = train
        self._test = test
        self._average_weight = 0
        self._average_rank = 0
        self._train_test_model()
        

    def _train_test_model(self):
        # item.details is in json format. Load them into a new features field
        for item in self._train:
            item.features = json.loads(item.details)
        for item in self._test:
            item.features = json.loads(item.details)
        # Some items don't have weight or rank feature
        # We use train data to calculate the average_weight and average_rank
        # We will apply the average to both train and test data for consistency and prevent data leakage
        self._average_weight = self._get_avg_weight_from_train()
        self._average_rank = self._get_avg_rank_from_train()
        # Create df for model.fit and model.predict
        train_df = self._list_to_dataframe(self._train)
        test_df = self._list_to_dataframe(self._test[:250])

        # First we need to separate features and target
        feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']
        X_train = train_df[feature_columns]
        y_train = train_df["price"]
        X_test = test_df[feature_columns]
        
        y_test = test_df["price"]

        # Train a Linear Regression
        self._model.fit(X_train, y_train)
        # Print out training stats
        for feature, coef in zip(feature_columns, self._model.coef_):
            print(f"{feature}: {coef}")
        print(f"Intercept: {self._model.intercept_}")

        # Predict the test set and evaluate
        y_pred = self._model.predict(X_test)
        # Calculate errors
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")

    # data contains a list of items
    def run(self, data):
        Tester.test(self._linear_regression_pricer, data)

    # Test on individual item
    def _linear_regression_pricer(self, item):
        features = self._get_features(item)
        df = pd.DataFrame([features])
        return self._model.predict(df)[0]

    
    def _list_to_dataframe(self, items):
        """
        A utility function to convert our features into a pandas dataframe
        df will be useful when we do model.fit and model.predict
        """
        features = [self._get_features(item) for item in items]
        df = pd.DataFrame(features)
        # Add a price column used for the target
        df['price'] = [item.price for item in items]
        return df


    def _get_avg_weight_from_train(self):
        weights = [self._get_weight(item) for item in self._train if item.features.get('Item Weight')]
        return sum(weights) / len(weights)

    def _get_avg_rank_from_train(self):
        ranks = [self._get_rank(item) for item in self._train if item.features.get("Best Sellers Rank")]
        return sum(ranks) / len(ranks)

    def _get_features(self, item):
        """
        Prepare a list of features for each item.
        The features will be used for the model to train
        """
        return {
            "weight": self._get_weight_with_default(item),
            "rank": self._get_rank_with_default(item),
            "text_length": self._get_text_length(item),
            "is_top_electronics_brand": 1 if self._is_top_electronics_brand(item) else 0
        }

    def _get_rank(self, item):
        rank_dict = item.features.get("Best Sellers Rank")
        if rank_dict:
            ranks = rank_dict.values()
            return sum(ranks)/len(ranks)
        return None

    def _get_rank_with_default(self, item):
        """
        Calculate rank feature for each item
        """
        rank = self._get_rank(item)
        return rank or self._average_rank

    def _get_weight(self, item):
        """
        Hacky codes to normalize different weight units
        """
        weight_str = item.features.get('Item Weight')
        if weight_str:
            parts = weight_str.split(' ')
            amount = float(parts[0])
            unit = parts[1].lower()
            if unit=="pounds":
                return amount
            elif unit=="ounces":
                return amount / 16
            elif unit=="grams":
                return amount / 453.592
            elif unit=="milligrams":
                return amount / 453592
            elif unit=="kilograms":
                return amount / 0.453592
            elif unit=="hundredths" and parts[2].lower()=="pounds":
                return amount / 100
            else:
                print(weight_str)
        return None

    def _get_weight_with_default(self, item):
        """
        Calculate weight feature for each item
        """
        weight = self._get_weight(item)
        return weight or self._average_weight

    def _get_text_length(self, item):
        """
        Calculate text_length feature for each item
        """
        return len(item.test_prompt())

    def _is_top_electronics_brand(self, item):
        """
        Calculate is_top_electronics_brand feature for each item
        """
        brand = item.features.get("Brand")
        return brand and brand.lower() in TOP_ELECTRONICS_BRANDS


# Bag of Words Model
class BagOfWordsModel:

    """
    Step 1: use CountVectorizer to fit and transform training documents into X_train
        Each row is a document. Each column corresponds to a specific word in the document and value is the count
    Step 2: Use linear regression to train X_train, training prices
    Step 3: use the same CountVectorizer to transform test documents into X_test
    Step 4: Use linear regression to predict X_test on test prices
    """
    def __init__(self, train, test):
        np.random.seed(42)
        self._train = train
        self._test = test
        self._model = LinearRegression()
        # Used to turn documents into columns of words (features) and rows of # of occurences
        # Capped at 1000 features at most and ignore common English words
        self._vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self._train_test_model()

    def _train_test_model(self):
        # Use test_prompt() which doesn't have price
        train_documents = [item.test_prompt() for item in self._train]
        train_prices = [item.price for item in self._train]
        test_documents = [item.test_prompt() for item in self._test[:250]]
        test_prices = [item.price for item in self._test[:250]]

        # Train
        X_train = self._vectorizer.fit_transform(train_documents)
        self._model.fit(X_train, train_prices)
        # Check overall stats. There are still feature names which are words names here
        word_names = self._vectorizer.get_feature_names_out()
        coefficients = self._model.coef_
        # Get indices of top 10 coefficients by absolute value
        top_indices = np.argsort(np.abs(coefficients))[-10:]
        # Print top 10 feature coefficients
        for idx in top_indices:
            print(f"{word_names[idx]}: {coefficients[idx]:.4f}")

        X_test = self._vectorizer.transform(test_documents)
        y_pred = self._model.predict(X_test)
        mse = mean_squared_error(test_prices, y_pred)
        r2 = r2_score(test_prices, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")

    # Test on individual item
    def _bow_lr_pricer(self, item):
        test_doc = [item.test_prompt()]
        X_test = self._vectorizer.transform(test_doc)
        return max(self._model.predict(X_test)[0], 0)

    def run(self, data):
        Tester.test(self._bow_lr_pricer, data)

class Word2VecModel:
    """
    Step 1: Feed all the trained documents (training data test_prompt) into w2v
        The output is each row is a word and columns are dimensions of numbers
    Step 2: Use mean pooling to generate a single fixed-size vector for each training document.
        Together, it forms a feature matrix, each row is one document, columns are dimensions of numbers
    Step 3: Use linear regression model to train the feature matrix against prices
    Step 4: For each test document, get the w2v embedding from trained w2v
    Step 5: Use linear regression model to predict test document's prices
    """
    def __init__(self, train, test):
        np.random.seed(42)
        self._train = train
        self._test = test
        self._lr_model = LinearRegression()
        self._linear_svr_model = LinearSVR()
        # Use n_estimators=8 to spin off 8 decision tree models for fast training
        # Use max_depth=8 for fast training. Use n_jobs=8 for 8 CPU processing
        # Use verbose=2 to print out debugging messages
        self._random_forest_model = RandomForestRegressor(n_estimators=8, max_depth=8, random_state=42, n_jobs=8,
                                                          verbose=2)
        self._w2v_model = None
        self._train_test_model()


    def _plot_w2v_embeddings_in_3d(self):
        # verbose=1 prints out progress
        tsne = TSNE(n_components=3, verbose=1, random_state=42)
        # Reduce dimensions to 3D using t-SNE
        # It has 3 columns each maps to x, y, z coordinates. Each row is a word
        # Only show top 1000 most frequent words because the whole embeddings are more than 192k x 400 which is too big
        top_1000_words = self._w2v_model.wv.index_to_key[:1000]
        top_1000_word_vectors = np.array([self._w2v_model.wv[word] for word in top_1000_words])
        word_vectors_3d = tsne.fit_transform(top_1000_word_vectors)
        # Plotly 3D visualization
        fig = go.Figure(data=[go.Scatter3d(
            x=word_vectors_3d[:, 0],
            y=word_vectors_3d[:, 1],
            z=word_vectors_3d[:, 2],
            mode='markers+text',
            text=top_1000_words,
            marker=dict(
                size=8,
                color='blue',     # Set color to blue for better visibility
                opacity=0.8,
                line=dict(width=1, color='black')
            )
        )])
        # Update layout
        fig.update_layout(
            title="3D Top 1000 Word Embeddings Visualization",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            showlegend=False
        )
        # This will open a new tab in a browser and show the 3d graph
        fig.show()

    def _mean_pooling_document(self, document):
        words = simple_preprocess(document)
        words_vector = [self._w2v_model.wv[word] for word in words if word in self._w2v_model.wv]
        # Take mean across all rows (axis = 0)
        return np.mean(words_vector, axis=0) if words_vector else np.zero(self._w2v_model.vector_size)

    def _train_test_model(self, plot_w2v_embeddings=False):
        # Define a callback for Word2Vec to show a progress bar
        class TqdmWord2Vec(CallbackAny2Vec):
            def __init__(self, epochs):
                self.epochs = epochs
                self.pbar = tqdm(total=self.epochs, desc="Training Word2Vec", unit="epoch")
    
            def on_epoch_end(self, model):
                self.pbar.update(1)  # Update progress bar after each epoch

            def on_train_end(self, model):
                self.pbar.close()  # Close progress bar when training is complete

        
        # Use test_prompt() which doesn't have price
        train_documents = [item.test_prompt() for item in self._train]
        train_prices = [item.price for item in self._train]
        test_documents = [item.test_prompt() for item in self._test[:250]]
        test_prices = [item.price for item in self._test[:250]]
        # Preprocess each document into tokens and wrap in tqdm to show progress bar
        processed_trained_docs = [simple_preprocess(doc) for doc in tqdm(train_documents,
                                                                         desc="Tokenizing Train Documents")]
        # Train Word2Vec model
        # Each word will have 400 dimensions (vectors)
        # look ahead and look back window size is 5
        # Include each word that occurs at least 1 time
        # Parallel processing to 8 CPUs
        epochs = 5
        self._w2v_model = Word2Vec(sentences=processed_trained_docs, vector_size=400, window=5, min_count=1,
                                   workers=8,
                                   epochs=epochs,
                                   callbacks=[TqdmWord2Vec(epochs)]  # Attach progress callback
                                   )
        # Extract word vectors and words
        words = self._w2v_model.wv.index_to_key
        word_vectors = np.array([self._w2v_model.wv[word] for word in words])

        if plot_w2v_embeddings:
            self._plot_w2v_embeddings_in_3d()

        X_train = np.array([self._mean_pooling_document(document) for document in tqdm(train_documents,
                                                                                       desc="Mean pooling training docs")])
        X_test = np.array([self._mean_pooling_document(document) for document in tqdm(test_documents,
                                                                                      desc="Mean pooling testing docs")])

        # Linear Regression Model
        self._train_and_test(self._lr_model, X_train, train_prices, X_test, test_prices)

        # Linear SVR model
        self._train_and_test(self._linear_svr_model, X_train, train_prices, X_test, test_prices)

        # RandomForest model
        self._train_and_test(self._random_forest_model, X_train, train_prices, X_test, test_prices)

    def _train_and_test(self, model, X_train, train_prices, X_test, test_prices):
        # Train
        model.fit(X_train, train_prices)

        # Testing
        y_pred = model.predict(X_test)
        mse = mean_squared_error(test_prices, y_pred)
        r2 = r2_score(test_prices, y_pred)
        print(f"{model} - Mean Squared Error: {mse}")
        print(f"{model} - R-squared Score: {r2}\n")


    # Test on individual item
    # Use random forest as default
    def _w2v_pricer(self, item):
        X_test = self._mean_pooling_document(item.test_prompt())
        # Need to wrap the 1D np array in [] to turn to 2D np array which is required for model.predict
        return max(self._random_forest_model.predict([X_test])[0], 0)

    def run(self, data):
        Tester.test(self._w2v_pricer, data)
        
    
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Choose a price model")
    # Add --model argument
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Price model to use"
    )
    # Parse the arguments
    args = parser.parse_args()
    # Convert the model name to lowercase
    model_name = args.model.lower()
    # Validate model_name
    if model_name not in MODEL_CLASSES:
        print(f"Invalid model name. Choose from: {', '.join(MODEL_CLASSES)}")
        sys.exit(1)

    # load in training data
    with open('train.pkl', 'rb') as file:
        train = pickle.load(file)

    # load in testing data
    with open('test.pkl', 'rb') as file:
        test = pickle.load(file)
    
    # Instantiate the appropriate model class and run it
    if model_name == 'random':
        model = RamdonPriceModel()
        # Predict prices on test data
        model.run(test)
    elif model_name == 'constant':
        model = ConstantPriceModel(train)
        # Predict prices on test data
        model.run(test)
    elif model_name == 'linear_regression':
        model = LinearRegressionModel(train, test)
        model.run(test)
    elif model_name == 'bag_of_words':
        model = BagOfWordsModel(train, test)
        model.run(test)
    elif model_name == 'word2vec':
        model = Word2VecModel(train, test)
        model.run(test)
    else:
        print(f"Invalid model name. Choose from: {', '.join(MODEL_CLASSES)}")
        sys.exit(1)
    sys.exit(0)  # Exit with zero status indicating success
