import gensim
import time
import yaml
import numpy as np
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, KeywordsOptions, SentimentOptions, EmotionOptions
from flask import Flask, jsonify, request
from text_analysis import get_emotion, get_keywords, get_sentiment

def phrase_mean_vectors(model, words):
    words_vectors = np.array([model[x] for x in words.split() if x in model])
    return np.mean(words_vectors, axis=0)

def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)

def append_text(compiled, text, max_words):
    compiled = compiled.split()
    text = text.split()
    compiled.extend(text)
    if len(compiled) > max_words:
        return ' '.join(compiled[-max_words:])
    else:
        return ' '.join(compiled)

app = Flask(__name__)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
keywords_thresh = config['ibm']['keywords-thresh']
categories = config['topic']['list']

### STATEFUL VAR
MAX_WORDS = config['ibm']['max_words']
compiled_path = "compiled.txt"
with open(compiled_path, 'w', encoding='utf8') as f:
    f.write("")

### IBM API INIT

authenticator = IAMAuthenticator(config['ibm']['api-key'])
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
natural_language_understanding.set_service_url(config['ibm']['api-url'])

### WORD2VEC MODEL

# start = time.time()
# model = gensim.models.KeyedVectors.load_word2vec_format(config['model']['path'], 
#                                                         binary=config['model']['binary'])
# print("Model loaded in {:.2f}s\n".format(time.time() - start))

### API

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    input_text = data['text']
    compiled_text = ""
    with open(compiled_path, 'r', encoding='utf8') as f:
        compiled_text = f.readline()
        
    new_text = compiled_text + " " + input_text
    print(new_text)
    
    response = natural_language_understanding.analyze(
        text=new_text,
        features=Features(sentiment=SentimentOptions(),
                          emotion=EmotionOptions(),
                          keywords=KeywordsOptions(sentiment=False,
                                                   emotion=False,
                                                   limit=config['ibm']['keywords-limit']))).get_result()
    
    res = {}
    
    with open(compiled_path, 'w', encoding='utf8') as f:
        f.write(input_text)
    
    # Keywords
    keywords = [x['text'] for x in response['keywords'] if x['relevance'] > keywords_thresh]
    print("Keyword candidates: {}".format(keywords))
    
    # categories_vectors = np.array([phrase_mean_vectors(model, words) for words in categories])
    # similarities = {x: softmax(np.linalg.norm(
    #                 (categories_vectors-phrase_mean_vectors(model,x)), axis=1))
    #                 for x in keywords}
    
    res['keywords'] = keywords
    
    # Emotion
    res['emotion'] = get_emotion(response['emotion']['document']['emotion'], 
                                 config['ibm']['emotion-thresh'])
    
    # Sentiment
    res['sentiment'] = get_sentiment(response['sentiment']['document']['score'], 
                                     config['ibm']['sentiment-thresh'])
    
    return jsonify(res)

if __name__ == '__main__':
    # app.run(debug=True)
    pass