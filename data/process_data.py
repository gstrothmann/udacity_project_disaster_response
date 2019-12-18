import sys
import pandas as pd
import nltk
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges the data from the two filepaths specified and saves this data into a pandas dataframe called df.    
    
    INPUT: None
    OUTPUT: df - pandas DataFrame
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''
    Cleans the data inside the dataframe df.
    The cleaning process consists of splitting the category column into seperate columns, renaming the new columns and removing duplicates.
    
    INPUT: df - pandas DataFrame
    OUTPUT: df - pandas DataFrame
    '''
    category_colnames = df.categories.str.split('-[01];*', expand = True).iloc[0,:].values[:-1]
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';*[a-z_]+-', expand = True)
    categories.drop(columns = [0], inplace=True)
    
    for col in categories.columns:
        categories[col] = pd.to_numeric(categories[col])
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)
    
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates('message', inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Saves a dataframe into an sqlite database with a specified filename.
    
    INPUT: 
    df - pandas DataFrame
    database_filename - string
    
    OUTPUT: None
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessagesAndCategories', engine, index=False, if_exists = 'replace')


def main():
    
    '''
    Performs the steps Loading, Cleaning, and Saving data by calling the respective functions.
    Returns an error message if the needed input has not been provided or does not meet the criteria.
    
    INPUT:
    messages_filepath - string
    categories_filepath - string
    database_filepath - string
    OUTPUT: None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
