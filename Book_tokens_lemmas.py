import spacy
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files

regexp = re.compile('(?u)\\b\\w\\w+\\b')

en_nlp = spacy.load("en_core_web_trf", disable=['parser', 'ner'])

old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))

#stemmer = nltk.stem.PorterStemmer()

#def compare_normalization(doc):
#    doc_spacy = en_nlp(doc)
#    print('Лемманизация:')
#    print([token.lemma_ for token in doc_spacy])

#    print('Стемминг:')
#    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]

lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

reviews_train = load_files('C:/Users/Admin/Desktop/PythonProjects/aclImdb/train',
    categories=['pos', 'neg'])

text_train, y_train = reviews_train.data, reviews_train.target
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

lemma_vect.fit(text_train)
X_train_lemma = lemma_vect.transform(text_train)
print('Форма X_train_lemma: {}'.format(X_train_lemma.shape))

vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print('Форма X_train: {}'.format(X_train.shape))

#compare_normalization(u"Our meeting today was worse than yesterday, " 
#    "I'm scared of meeting the clients tomorrow.")