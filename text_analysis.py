import numpy as np

# print(json.dumps(response, indent=2))
### SENTIMENT EXTRACTION
def get_sentiment(sent_score, sent_thresh):
    sentiment = "neutral"
    if sent_score > sent_thresh:
        sentiment = "positive"
    elif sent_score < -sent_thresh:
        sentiment = "negative"
    print("Sentiment: {} ({:.2f})".format(sentiment, sent_score))
    print()
    
    return {"label": sentiment, "score": sent_score}

### EMOTION DETECTION
def get_emotion(emot_score, emot_thresh):
    emotion = max(emot_score, key=emot_score.get)
    if emot_score[emotion] <= emot_thresh:
        emotion = 'neutral'
    print("Emotion score: {}".format(emot_score))
    print("Emotion: {}".format(emotion))
    print()
    
    return emotion

### KEYWORDS EXTRACTION
def get_keywords(similarities, categories, e=0.3):
    topic_conf_thresh = (e-1)/(e*len(categories))
    
    res = {}
    for k in similarities.keys():
        min_dis = np.min(similarities[k])
        if min_dis < topic_conf_thresh:
            min_idx = np.argmin(similarities[k])
            res[k] = categories[min_idx]
            
    print("Keywords: {}".format(res))
    print()
    
    return list(res.keys())

if __name__ == '__main__':
    import gensim
    import time
    import yaml
    import json
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 \
        import Features, KeywordsOptions, SentimentOptions, EmotionOptions
        
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    keywords_thresh = config['ibm']['keywords-thresh']
    categories = config['topic']['list']
    
    with open('sample.txt', 'r', encoding='utf8') as f:
        input_text = f.readline()
    print(input_text)

    ### IBM API INIT

    authenticator = IAMAuthenticator(config['ibm']['api-key'])
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2022-04-07',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(config['ibm']['api-url'])

    ### WORD2VEC MODEL

    # start = time.time()
    # model = gensim.models.KeyedVectors.load_word2vec_format(config['model'], binary=True)
    # print("Model loaded in {:.2f}s\n".format(time.time() - start))
    
    response = natural_language_understanding.analyze(
        text=input_text,
        features=Features(sentiment=SentimentOptions(),
                          emotion=EmotionOptions(),
                          keywords=KeywordsOptions(sentiment=False,
                                                   emotion=False,
                                                   limit=config['ibm']['keywords-limit']))).get_result()
    
    print(json.dumps(response, indent=2))