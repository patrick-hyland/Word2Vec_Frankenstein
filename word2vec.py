filename = 'frankenstein_text.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

import gensim
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#import mpld3

corpus_raw = text.lower()


noise_list = ["is", "a", "this", "...","/n","the","I","you","at","in","an","of","to","chapter"]
stopwords.words('english').append(noise_list)
noise=stopwords.words('english').append(noise_list)

for i in stopwords.words('english'):
    noise_list.append(i)
noise_list=set(noise_list)

def _remove_noise(input_text):
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

cleaned=_remove_noise(corpus_raw)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def lemmer(text):
    words=text.split()
    new_words=[lem.lemmatize(word,"v") for word in words]
    new_text=" ".join(new_words)
    return new_text

def dropper(text):
    words=text.split()
    new_words=[word.replace('?','').replace('!','').replace(',','').replace(';','').replace('\"',"") for word in words]
    newer_words=[word for word in new_words if len(word)>2]
    new_text=" ".join(newer_words)
    return new_text

cleaned=lemmer(cleaned)
cleaned=dropper(cleaned)

raw_sentences = cleaned.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

model = gensim.models.Word2Vec(sentences, min_count=10,iter=20)

vocab = list(model.wv.vocab)
X = model[vocab]

normalizer = preprocessing.Normalizer()
X =  normalizer.fit_transform(X, 'l2')

tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)
df = pd.concat([pd.DataFrame(X_tsne), pd.Series(vocab)], axis=1)
df.columns = ['x','y','z','word']
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.scatter(df['x'], df['y'],df['z'])

for i, txt in enumerate(df['word']):
    ax.text(df['x'].iloc[i], df['y'].iloc[i],df['z'].iloc[i],txt)
    #print(i,txt)
plt.title("Frankenstein: Mary Shelley")

plt.show()
