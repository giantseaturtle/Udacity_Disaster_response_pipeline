import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Generate X, Y from loaded dataset

    Argument:
    database_filepath (string) -- filepath of loaded databaase

    Output:
    X, Y -- feature and labels Dataframe
    category_names -- list of category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterTable',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and lematize import text

    Argument:
    text (list) -- text to be cleaned

    Output:
    clean_tokens (list) -- cleaned text for build_model
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the starting verb of text, create a new feature for classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build pipeline to proecess text messages and apply classifier,
    gridsearch to find the best parameters for the model

    Output:
    tuned_model -- tuned Machine learning model
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10, 40],
        'clf__estimator__min_samples_leaf': [2, 5]
        }
    tuned_model = GridSearchCV(model,param_grid=parameters)
    return tuned_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the performance of model by showing precision, recall, f1-score

    Argument:
    model (pipeline model) -- trained model
    X_test -- test features
    Y_test -- test labels
    category_names -- label names to be evaluated
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print("Category: {}".format(col))
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    export model as a pickle file

    Argument:
    model -- trained model
    model_filepath -- destination of saved file
    """
    filename = 'classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
