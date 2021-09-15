import numpy as np
import keras
import string
import re

with open(r"D:\Programmes\MachLearn\generatif\texte\bible.txt", encoding='utf-8') as f :
     text = f.read()
     f.close()
     
text=text.replace("\n", " ")
text=text.lower()
text=re.sub(r'[0-9]+', '', text)
text=text.translate(str.maketrans("","", string.punctuation))
liste_mots = text.split()

#%% On garde les 5000 mots les plus fréquents
n_words=5000

words_dico = dict()
for word in liste_mots:
    if word not in words_dico:
        words_dico[word]=1
    else :
        words_dico[word]+=1
        
#conversion en liste pour tri
words_freq = list()
for key, val in words_dico.items():
    words_freq.append( (key, val) )
words_freq.sort(key=lambda tup: tup[1] ,reverse=True)
#print("Le mot le plus fréquent : ",words_freq[0])

#liste de mots à supprimer
stopwords = [t[0] for t in words_freq[n_words:]]

#suppression dans le texte
new_list = []
for i in range(len(liste_mots)):
    if liste_mots[i] not in stopwords:
        new_list.append(liste_mots[i])
liste_mots=new_list
        

#%%
maxlen, step = 40, 5

sentences, next_words = [], []

for i in range(0, (len(liste_mots)-maxlen)//7, step):
    sentences.append(liste_mots[i:i+maxlen])
    next_words.append(liste_mots[i+maxlen])
    
#%%
words = sorted((set(liste_mots)))

words_indices = dict((word, words.index(word)) for word in words)

X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, words_indices[word]]=1
    y[i, words_indices[next_words[i]]]=1
    

#%%
from keras import layers, models, optimizers, callbacks

model=models.Sequential()
model.add(layers.GRU(256, activation='relu', return_sequences=False, input_shape=(maxlen, len(words))))
#model.add(layers.GRU(128, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(words), activation="softmax"))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=optimizers.RMSprop(lr=0.01))

callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=2)]


model.fit(X, y, batch_size=64, epochs=60, validation_split=0.05, callbacks=callbacks)
model.save("bible.h5")
    

#%%
def sample(preds, temperature=1):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,size=1)
    return np.argmax(probas)


start = np.random.randint(0, len(liste_mots)-maxlen)
generated_text = liste_mots[start:start+maxlen]
test_text = liste_mots[start:start+maxlen]    

generated_text_string=""
for w in generated_text:
    generated_text_string+=w+" "
print("amorce : " + generated_text_string)

    
temperature=0.5

generated_text_string=""
for i in range(400):
    sampled=np.zeros((1,maxlen,len(words)))
    for t, word in enumerate(test_text):
        sampled[0,t,words_indices[word]]=1
    next_word=words[sample(model.predict(sampled, verbose=0)[0], temperature=temperature)]
    generated_text_string+=next_word+" "
    test_text=generated_text[:1]

print("\n resultat : " + generated_text_string)
    
    
#%%
X_test = np.zeros((1, maxlen, len(words)), dtype=np.bool)

num_test, temperature = 6, 0.1

for t, word in enumerate(sentences[num_test]):
    X_test[0, t, words_indices[word]]=1
    
preds=model.predict(X_test)[0]
next_word=words[sample(preds, temperature=temperature)]

print("amorce : "+sentences[num_test])
print("next character predicted : "+ next_word)
print("next character true : "+ next_words[num_test])
    
    
    
    
    
    
    
    
    
    
    
    
    
    