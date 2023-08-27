import json
import string

def slicing_splitting(raw_text):
    '''Slices book text from raw format and splits into chapter-indexed lists. No formatting changes.'''

    print('Slicing and splitting text into chapters...')
    start = 'CHAPTER I. Down the Rabbit-Hole'
    end = 'THE END'
    start_ind = raw_text.find(start)
    end_ind = raw_text.find(end)
    novel = raw_text[start_ind:end_ind+len(end)]
    split_text = novel.split("CHAPTER")

    chapter_list = []
    for chapter_text in split_text:
        if chapter_text.strip():
            chapter_list.append("CHAPTER" + chapter_text)

    with open('../data/chapter_list.json', 'w') as json_file:
        json.dump(chapter_list, json_file)
    print('Text sliced, split and saved as chapter_list.')

    return chapter_list

def extra_token_cleaning(list_of_tokens):
    '''Removing punctuation kept inside tokens aside from hyphens. Specifically aiming at words of the form '_very_' '''

    index = string.punctuation.find('-')
    punc_list = string.punctuation[:index] + string.punctuation[index+1:]

    result = []
    for token in list_of_tokens:
        if any(char.isalpha() for char in token):
            result.append(''.join([char for char in token if char.isalpha() or char == '-']))
    return result
