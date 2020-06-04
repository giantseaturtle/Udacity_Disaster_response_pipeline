import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load dataframe from filepath

    Argument:
        messages_filepath (string) -- path to the messages csv file
        categories_filepath (string) -- path to the catergories csv file

    Output:
        df(dataframe) -- Combined dataframe from messages and catergories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Clean "catergories" column in df

    Argument:
        df (dataframe) -- Combined dataframe from messages and catergories

    Output:
        df (dataframe) -- Cleaned dataframe
    """
    # split categories into separate category columns
    categories = df['categories'].str.split(';',expand=True)

    # name the separate category columns with first row of Dataframe
    row = categories.iloc[0, : ]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to numeric values
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop(columns=['categories'],axis=1)
    df = pd.concat([df, categories], axis=1)

    # drop the duplicates in df
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the clean dataframe into an sqlite database

    Argument:
        df (dataframe) -- Cleaned dataset
        database_filename (string) -- Sqlite destination file path
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterTable', engine, index=False) 


def main():
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
