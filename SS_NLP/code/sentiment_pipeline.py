import json
import os
from nltk import sent_tokenize
from modelling import predict_model, initialize_model

def load_sentiment_data():
    '''Loads and returns data for sentiment analysis from pre-determined paths.'''

    print('Loading sentiment data...')

    pwd =  os.getcwd()
    path1 = pwd + '/../data/chapter_list.json'
    with open(path1, "r") as json_file:
        chapter_list = json.load(json_file)

    chapter_sentence_list = []
    for chapter in chapter_list:
        chapter_sentence_list.append(sent_tokenize(chapter))

    path2 = pwd + '/../data/top_tdidf_terms_10_list.json'
    with open(path2, "r") as json_file:
        top_10 = json.load(json_file)
    print('Sentiment data loaded.')

    return chapter_sentence_list, top_10

def filter_by_key_terms(chapter_sentence_list, filter_list):
    '''Filters a nested list of sentences within chapters by key words (per chapter).'''

    print('Filtering sentences...')
    chapter_sentences_filtered = []
    for index, chapter in enumerate(chapter_sentence_list):
        top_10_words, top_10_scores = zip(*filter_list[index])
        filtered_sentences = [sentence for sentence in chapter if any(word in sentence.lower() for word in top_10_words)]
        chapter_sentences_filtered.append(filtered_sentences)
    print('Filtered sentences.')

    return chapter_sentences_filtered

def process_data(chapter_sentence_list, filter_list = None):
    '''Filters data if specified. Models the data and saves the model predictions to a JSON.'''

    print('Processing data...')

    if filter_list:
        chapter_sentence_list = filter_by_key_terms(chapter_sentence_list, filter_list)

    model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
    classifier = initialize_model(model_name)

    print('Modelling data...')
    chapter_predictions = []
    for index, chapter in enumerate(chapter_sentence_list):
        predictions = []
        # Able to inplement batching here for larger data throughput requirements. Requires implementing padding and masking.
        for sentence in chapter:
                prediction = predict_model(classifier, sentence)
                predictions.append([index, sentence, prediction])
        chapter_predictions.append(predictions)
        print(f'Chapter {index+1} finished modelling.')

    return chapter_predictions

if __name__ == "__main__":
    chapter_sentence_list, top_10 = load_sentiment_data()
    chapter_predictions = process_data(chapter_sentence_list = chapter_sentence_list, filter_list = top_10)

    path3 = '/home/peter/code/PeterLambertGitt/SS_NLP/data/chapter_predictions.json'
    with open(path3, "w") as json_file:
        json.dump(chapter_predictions, json_file)
