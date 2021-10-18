from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.decomposition import LatentDirichletAllocation

reviews_train = load_files('C:/Users/Admin/Desktop/PythonProjects/aclImdb/train',
    categories=['pos', 'neg'])

text_train, y_train = reviews_train.data, reviews_train.target
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)

lda = LatentDirichletAllocation(n_components=10, learning_method='batch',
    max_iter=25, random_state=0)

document_topics = lda.fit_transform(X)

print(lda.components_.shape)