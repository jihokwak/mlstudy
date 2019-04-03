import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
nltk.download('all')

term = 'present'

synsets = wn.synsets(term)
type(synsets)
len(synsets)
for synset in synsets :
    print("POS : ", synset.lexname())
    print("Definition :", synset.definition())
    print("Lemma : ", synset.lemma_names())

tree = wn.synset('tree.n.01')
lion = wn.synset('lion.n.01')
tiger = wn.synset('tiger.n.02')
cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')

entities = [tree, lion, tiger, cat, dog]
similarities = []
entity_names = [entity.name().split('.')[0] for entity in entities]

for entity in entities :
    similarity = [round(entity.path_similarity(compared_entity), 2) for compared_entity in entities]
    similarities.append(similarity)

similarity_df = pd.DataFrame(similarities, columns=entity_names, index=entity_names)
similarity_df

#SentiWordNet
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

def penn_to_wn(tag) :
    if tag.startswith("J") :
        return wn.ADJ
    elif tag.startswith("N") :
        return wn.NOUN
    elif tag.startswith("R") :
        return wn.ADV
    elif tag.startswith("V") :
        return wn.VERB

from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag

def swn_polarity(text) :
    sentiment = 0.0
    tokens_count = 0

    lemmatizer = WordNetLemmatizer()
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences :
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma :
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets :
                continue
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += (swn_synset.pos_score() - swn_synset.neg_score())
            tokens_count += 1
    if not tokens_count :
        return 0

    if sentiment >= 0:
        return 1

    return 0


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy , precision ,recall))


import pandas as pd
import re
review_df = pd.read_csv('data/labeledTrainData.tsv', header=0, sep="\t", quoting=3)
review_df['review'] = review_df['review'].str.replace('<br />',' ')
review_df['review'] = review_df['review'].apply( lambda x : re.sub("[^a-zA-Z]", " ", x) )

review_df['preds'] = review_df['review'].apply(lambda x : swn_polarity(x))
y_target = review_df['sentiment'].values
preds = review_df['preds'].values

print("***예측성능평가***")
get_clf_eval(y_target, preds)

#VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analyzer = SentimentIntensityAnalyzer()
senti_scores = senti_analyzer.polarity_scores(review_df['review'][0])
senti_scores

def vader_polarity(review, threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)

    agg_score = scores['compound']
    final_sentiment = 1 if agg_score >= threshold else 0
    return final_sentiment

review_df['vader_preds'] = review_df['review'].apply(lambda x : vader_polarity(x, 0.1))
y_target = review_df['sentiment'].values
vader_preds = review_df['vader_preds'].values

print("### VADER 예측 성능 평가 ###")
get_clf_eval(y_target, vader_preds)