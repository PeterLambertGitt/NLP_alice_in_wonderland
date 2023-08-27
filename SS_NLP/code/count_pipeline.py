import json
import pandas as pd
from nltk.tokenize import word_tokenize
import os
from preprocessing import slicing_splitting, extra_token_cleaning

def load_raw_text(path_ext: str):
    '''Loads and returns the raw text data from predefined path.'''

    print('Loading raw text...')

    pwd =  os.getcwd()
    path = pwd + path_ext
    f = open(path, "r", newline = None)
    raw_text = ' '.join(f.read().splitlines())

    print('Raw text loaded.')
    return raw_text

def count_pipeline(raw_text: str):
    '''Takes raw text as an input, pre-processes, then saves both tokenized chapters in a list and words counted by chapter in a dataframe.'''

    print('Preprocessing and tokenizing text...')

    chapter_list = slicing_splitting(raw_text)

    with open('../data/chapter_list.json', 'w') as json_file:
        json.dump(chapter_list, json_file)

    print('Split chapters saved under chapter_list')

    chapters_tokenized = []

    for chapter in chapter_list:
        chapters_tokenized.append(extra_token_cleaning(word_tokenize(chapter.lower())))

    with open('../data/chapters_tokenized.json', 'w') as json_file:
        json.dump(chapters_tokenized, json_file)

    print('Preprocessed and tokenized text saved under chapters_tokenized')

    chapter_vocab = []
    for chapter in chapters_tokenized:
        chapter_vocab.append(list(set((chapter))))

    df_info = []
    for i in range(0, 12):
        for word in chapter_vocab[i]:
            chapter_number = i+1
            df_info.append((chapter_number, word))

    columns = ['chapter_number', 'word']
    count_df = pd.DataFrame(df_info, columns=columns)

    def calculate_count(row):
        '''Calculates word count by chapter number and word.'''

        chapter_number = row['chapter_number']
        word = row['word']
        chapter_words = chapters_tokenized[chapter_number-1]
        number_of_word = chapter_words.count(word)
        return int(number_of_word)

    count_df['count'] = count_df.apply(calculate_count, axis=1)

    check_count(count_df,chapters_tokenized)

    count_df.to_csv('../data/wordcount_df.csv')
    print('Word count dataframe saved.')

def check_count(count_df, chapters_tokenized: list):
    '''Provides error checking for dataframe. Checks that all words are counted and none are counted twice.'''

    total_words = []
    for chapter in chapters_tokenized:
        for word in chapter:
            total_words.append(word)

    remaining_word_list = total_words

    for _, row in count_df.iterrows():
        count = row['count']
        word = row['word']
        for i in range(count):
            remaining_word_list.remove(word)

    assert remaining_word_list == [], f'Error. Words not counted in dataframe are: {remaining_word_list}'

if __name__ == "__main__":
    raw_text = load_raw_text(path_ext = '/../data/alice_in_wonderland.txt')
    count_pipeline(raw_text)
