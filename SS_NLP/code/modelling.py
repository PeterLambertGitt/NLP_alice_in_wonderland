from transformers import pipeline

def initialize_model(model_name: str):
    '''Load the model along with it's inbuilt tokenizer and embeddings. Needs to be seperated from prediction to stop the model being loaded repetitively'''

    print('Loading model...')
    classifier = pipeline("text-classification",model=model_name, top_k=None)
    print('Model loaded.')

    return classifier

def predict_model(classifier, sentence):
    '''Processes data through the model and outputs predictions'''

    prediction = classifier(sentence)

    return prediction
