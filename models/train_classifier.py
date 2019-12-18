import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Loads data from the specified database filepath and creates the variables X and Y, containing the features and labels of the training data. 
    It also creates a list of category names.
    
    INPUT: database_filepath - string
    OUTPUT:
    X - list of strings
    Y - pandas DataFrame
    category_names - list of strings
    
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesAndCategories', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df[df.columns[4:]].columns)
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes the provided text by removing punctuation, setting all letters to lower case, separating all words, removing stopwords and lemmatizing the words.   
    
    INPUT: text - string
    OUTPUT: lemmed - list of strings
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    '''
    Builds a model pipeline that consists of a CountVectorizer, using the tokenizer function as a tokenizer, a TfidfTransforemer and a Multioutputclassifier function. 
    The function uses GridSearchCV in order to find the best parameters of the used classifier (RandomForestClassifier) as well as the TFidFTransformer.
        
    INPUT: None
    OUTPUT: model - GridSearchCV object
    '''
    pipeline = Pipeline([
    ('vec', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'clf__estimator__min_samples_leaf':[1, 2],
    'clf__estimator__min_samples_split':[2,3,5],
    'clf__estimator__criterion':['gini', 'entropy'],
    'tfidf__use_idf':[True, False], 
    }

    model = GridSearchCV(pipeline, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Makes predictions with the pretrained model, using passed test data, and prints out classification reports for each label within the dataset.     
    
    INPUT:
    model - classification model
    X_test - list of lists with tokenized messages
    Y_test - list of classification labels
    category_names - list of strings
    
    OUTPUT: None
    '''
    
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Saves model to a pickle file with the specified filepath.
    
    INPUT: 
    model - sklearn model
    model_filepath - string
    
    OUTPUT: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Proceeds the steps of Building, Training, Evaluating and Saving a model.
    
    INPUT: None
    OUTPUT: None
    '''
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
