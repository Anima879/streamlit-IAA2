from typing import List, Union
import textblob
import pickle
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
import re
import contractions

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

with open('model', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,
                'S': wordnet.ADJ_SAT}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatized(token: str):
    return lemmatizer.lemmatize(token, get_wordnet_pos(token))


regex = r'\b(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)\b[\w\s]+[^\w\s]'
sub_regex = r'\1NEG_\2'

tokenizer = TreebankWordTokenizer()
ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize(x: str):
    # First, we remove upper cases and contractions (I've done the work => i have done the work)
    x = contractions.fix(x.lower())
    # Remove ponctuation and special char but we keep '$' symbol because it can be meaningful.
    x = re.sub('[^A-Za-z0-9$ ]+', '', x).strip()
    x = " ".join(x.split())
    # Adding the "neg_" prefix
    transformed = re.sub(regex,
                         lambda match: re.sub(r'(\s+)(\w+)', sub_regex, match.group(0)),
                         x,
                         flags=re.IGNORECASE)
    # Tokenization
    tokens = tokenizer.tokenize(transformed)
    # Removing stop words and stemming.
    tokens = [lemmatized(x) for x in tokens if x not in stop_words]
    sentence = str(' '.join(tokens))
    return sentence


topic_dict = {
    "Topic 0": "Lieu et période de visite",
    "Topic 1": "Autre",
    "Topic 2": "Goût et saveurs",
    "Topic 3": "Livraison",
    "Topic 4": "Pizza",
    "Topic 5": "Prix et qualité du service",
    "Topic 6": "Service à table et bar",
    "Topic 7": "Temps de service",
    "Topic 8": "Burger et frites",
    "Topic 9": 'Décoration',
    "Topic 10": "Viande",
    "Topic 11": "Bar et boissons",
    "Topic 12": "Relations clients",
    "Topic 13": "Experiences",
    "Topic 14": "Rapport qualité-prix",
    "Topic 15": "Sushi et japonais"
}


def get_topic_by_idx(idx: int) -> str:
    return topic_dict[f"Topic {idx}"]


def get_n_argmax(n: int, _values: list) -> list:
    values = [x for x in _values]
    arg_max = []
    minimum = min(values)
    for _ in range(n):
        a = np.argmax(values)
        arg_max.append(a)
        values[a] = minimum

    return arg_max


def prediction(text: str, n_topics: int = 1, polarity_threshold=.3) -> Union[List[str], str]:
    if n_topics < 1 or n_topics > 15:
        raise ValueError("number of topics must be between 1 and 15")

    blob = textblob.TextBlob(text)

    polarity = blob.sentiment.polarity
    if polarity < polarity_threshold:
        normalized_text = normalize(text)
        vec = vectorizer.transform([normalized_text])
        result = model.transform(vec)[0]
        topics_idx = get_n_argmax(n_topics, result)
        topics = []
        for i in topics_idx:
            topics.append(get_topic_by_idx(i))
        return topics
    else:
        return f"The text is not a negative sentiment (polarity={polarity})"
