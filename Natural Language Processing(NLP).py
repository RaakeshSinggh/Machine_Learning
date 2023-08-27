#!/usr/bin/env python
# coding: utf-8

# In[10]:


structured data

table format
csv file, excel files
rdbms
rows and columns


unstructured data

text file
image
audio/video
signal data 


# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[2]:


data = load_iris()


# In[3]:


data['target_names']


# In[4]:


data.keys()


# In[5]:


data['target']


# In[6]:


data


# In[7]:


data['feature_names']


# In[8]:


df=pd.DataFrame(data.data,columns = data.feature_names)


# In[9]:


df


# In[10]:


df['output']=data['target']


# In[11]:


df


# In[12]:


df['lable_names']=df['output'].map({0:'setosa',1:'versicolor',2:'virginica'})


# In[13]:


df


#  document or corpus

# In[14]:


x = "i am attached with edyoda data science program. haksdhdddhdiuqdm   ,calcjkcmcc"


# In[15]:


reviews = ["food good, staff bad",
           "awsome",
           "locality is not appropriate for couples and fmily",
           "food is spicy, staff is ok!!!!",
           "i enjoyed my day with my friends",
           "khatarnak hai bro",
           "AWESOME",
           "superrrrrrrrrrrrrrrr se upperrrrrrrrrrrrr",
           "emojis",
           ]*1000


#  challenges in nlp

# In[20]:


extracting meaningful words from ambiguous, disorganized language
spelling mistakes
hinglish
data cleaning 
  special chracters
  text standardization
  character repition multiple times
  unnecessary comments
  numbers


# In[16]:


len(reviews)


# In[17]:


reviews


# In[18]:


l=[1,2]


# In[19]:


l*10


# In[20]:


reviews = ["food good, staff bad",
           "awsome",
           "locality is not appropriate for couples and fmily",
           "food is spicy, staff is ok!!!!",
           "i enjoyed my day with my friends",
           "khatarnak hai bro",
           "AWESOME",
           "superrrrrrrrrrrrrrrr se upperrrrrrrrrrrrr",
           "emojis",
           ]


# In[28]:


# 2 challenges
1. clean text
2. transform the text in such a way so that machine can process


# In[ ]:


# data cleaning

1. stop words removal 
frequency more, contribution very less in terms extracting the information from text 
a, an , the, i, me, him, is, am, are, 


2. Stemming and lemmatization

inflected word = root word + suffixes, prefixes + root word


going = go + ing
visited = visit + ed
premature =  pre + mature
immature = im + mature
unfit = un + fit 

stemming = extracting root word from inflected word based on their rules, root word ---> stem


lemmatization = extracting root word from inflected word based on their rules, root word ---> lemma


3. Tokenization = process of splitting doc/corpus into chunks(small part)
    word Tokenization
    senetence Tokenization=["food good, staff bad",
           "awsome",
           "locality is not appropriate. for couples and fmily",
           "food is spicy, staff is ok!!!!",
           "i enjoyed my day with my friends",
           "khatarnak hai bro",
           "AWESOME",
           "superrrrrrrrrrrrrrrr se upperrrrrrrrrrrrr",
           "emojis",
           ]



# N-gram = list of all n words in sentance

# In[28]:


x=['i am enjoing this class']
n = 1 = unigram = ['i, am, enjoing, this, class']
n = 2 = bigram = ['1 am, am enjoing, enjoing this, this class']
n = 3 = trigram = ['i am enjoing, am enjoing this, enjoing this class']


#  Words sense Disambiguation

# In[ ]:


process of identifying the correct meaning of the ambiguous word write sentence
Disambiguation = Dis + ambiguation
"i am enjoing this class"


# In[21]:


import nltk
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[22]:


df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")


# In[23]:


df = df[['v2', 'v1']]
df.columns = ['message', 'label']


# In[24]:


df


# In[25]:


message = df['message'].values


# In[26]:


len(message)


# In[27]:


message[:2]


# 1. Text standardization

# In[28]:


message_new = [m.lower() for m in message]


# In[29]:


Message_New = [m.upper() for m in message]


# In[30]:


Message_New[:2]


# In[31]:


message_new[:2]


# 2. removing special characters
# 3. maketrans, regex, replace() by passing custom mapping

# In[32]:


import re
message_re = [re.sub('\W',' ', m) for m in message_new]
message_re = [re.sub('\s+',' ', m).strip() for m in message_re]
# \w = [a-zA-Z0-9_]


# In[33]:


message_re[:2]


# In[34]:


import nltk
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')


# In[35]:


sw[:3]


# In[36]:


message_re[:2]


# In[37]:


# messages_without_sw = [[m] for msg in messages_re for m in msg.split() if m not in sw]
messages_without_sw = []
for msg in message_re:
    msg_cleaned = []
    for m in msg.split(): # msg.split()= word_tokenize(msg)
        if m not in sw:
            msg_cleaned.append(m)
    messages_without_sw.append(" ".join(msg_cleaned))


# In[38]:


messages_without_sw[:2]


# Droping numbers and single character

# In[39]:


messages_without_sw = [re.sub('\d+', '', m) for m in messages_without_sw]
messages_without_sw = [re.sub('\s\w{1,2}\s', '', m) for m in messages_without_sw]
messages_without_sw = [re.sub('^\w{1,2}\s', '', m) for m in messages_without_sw]
messages_without_sw = [re.sub('\s\w{1,2}$', '', m) for m in messages_without_sw]
messages_without_sw = [re.sub('\s+', ' ', m).strip() for m in messages_without_sw]


# In[40]:


messages_without_sw[:5]


# In[41]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize


# In[42]:


word_tokenize('i like study')


# In[43]:


sent_tokenize('i like study')


# In[44]:


word_tokenize("i like study. i don't like late night study.")


# In[45]:


word_tokenize("i like study. i don't like late night study.")


# In[46]:


sent_tokenize("i like study. i don't like late night study.")


# In[47]:


help(sent_tokenize)


# In[48]:


import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import PorterStemmer,WordNetLemmatizer
ps = PorterStemmer()
lm = WordNetLemmatizer()


# In[49]:


messages_stem = []
messages_lemma = []
for msg in messages_without_sw:
    msg_stem = []
    msg_lm = []
    for m in word_tokenize(msg):
        msg_stem.append(ps.stem(m))
        msg_lm.append(lm.lemmatize(m))
    messages_stem.append(" ".join(msg_stem))
    messages_lemma.append(" ".join(msg_lm))
    


# In[50]:


for m0, m1, m2 in zip(messages_without_sw[:5],messages_stem[:5],messages_lemma[:5]):
    print(m0)
    print(m1)
    print(m2)
    print('================================================')


# In[51]:


messages_stem = []
for msg in messages_without_sw:
    msg_stem = []
    for m in word_tokenize(msg):
        msg_stem.append(ps.stem(m))
    messages_stem.append(" ".join(msg_stem))


# In[52]:


messages_stem_new = []
for msg in messages_without_sw:
    messages_stem_new.append(" ".join([ps.stem(m) for m in word_tokenize(msg)]))


# In[53]:


messages_stem_new[:5]


# In[54]:


from nltk.wsd import lesk


# In[55]:


c1 = lesk(word_tokenize("cricket is a game of ball and bat"), 'bat')


# In[56]:


print(c1,c1.definition())


# In[65]:


c2 = lesk(word_tokenize("bat is ok at night"),'bat')


# In[66]:


print(c2,c2.definition())


# In[67]:


c1 = lesk(word_tokenize("i enjoy this class"),'class')
print(c1,c1.definition())


# In[69]:


c1 = lesk(word_tokenize("this class of product is awesome"), 'class')
print(c1, c1.definition())


# In[70]:


c1 = lesk(word_tokenize("face mask"), 'face')
print(c1, c1.definition())


# In[71]:


c1 = lesk(word_tokenize("face ofman"), 'face')
print(c1, c1.definition())


# In[72]:


messages_stem_new[:5]


# In[73]:


s = "jurong point crazi avail bugisgreat world buffet cine got amor wat"
s = word_tokenize(s)


# In[74]:


def get_n_grams(s,n):
    return[" ".join(s[i:i+n]) for i in range (len(s)-(n-1))]


# In[75]:


s = 'jurong point crazi avail bugisgreat world buffet cine got amor wat'
get_n_grams(word_tokenize(s), 3)


# In[76]:


from nltk.util import ngrams
n = list(ngrams(word_tokenize(s),6))
print(n,len(n))


# In[80]:


# transform the text in such a way so that machine can process
1. Bag of words model : take all the words in your corpus (0/1)
2. Count Vectorization (0/frequency)
3. Tf Idf Vectorization (0/tf*idf)
4. Hashing Based Vectorization


# In[77]:


x = ['i am watching a nice movie. movie is related to fiction. generally, i dont like movies.', 
     'students are enjoying outside of the class. they dont want to attend the class']


# In[78]:


bow = [y for m in x for y in m.split()]


# In[79]:


list(set(bow))


# In[80]:


messages_stem_new[:2]


# In[86]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[87]:


CountVectorizer().fit(['lar joke wifoni']).vocabulary_


# In[88]:


b_vect = CountVectorizer(max_features=20,binary=True)
c_vect = CountVectorizer(max_features=20)
t_vect = TfidfVectorizer(max_features=20)
h_vect = HashingVectorizer(n_features=20)


# In[90]:


c_vect.fit(messages_stem_new)


# In[92]:


len(c_vect.vocabulary_)


# In[94]:


messages_count_vect = c_vect.transform(messages_stem_new)


# In[96]:


print(messages_count_vect)


# In[97]:


x = messages_count_vect.toarray()


# In[98]:


y = df['label'].map({'ham':0, 'spam':1})


# In[99]:


from sklearn.model_selection import cross_val_score


# In[101]:


scores = cross_val_score(LogisticRegression(), x, y, cv = 10)


# In[103]:


scores.mean()


# In[104]:


# Binary Vectorization
messages_binary_vect = b_vect.fit_transform(messages_stem_new)
# Count Vectorization
messages_count_vect = c_vect.fit_transform(messages_stem_new)
# Tfidf Vectorization
messages_tfidf_vect = t_vect.fit_transform(messages_stem_new)
# Hasing Vectorization
messages_hasing_vect = h_vect.fit_transform(messages_stem_new)


# In[108]:


# h_vect.vocabulary_
messages_binary_vect.toarray()


# In[110]:


from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score


# In[121]:


b_scores = cross_val_score(BernoulliNB(), messages_binary_vect.toarray(), y, cv = 10)


# In[122]:


b_scores.mean()


# In[111]:


b_scores = cross_val_score(MultinomialNB(), messages_binary_vect.toarray(), y, cv = 10)


# In[125]:


b_scores.mean()


# In[123]:


c_scores = cross_val_score(MultinomialNB(), messages_count_vect.toarray(), y, cv = 10)


# In[124]:


c_scores.mean()


# In[115]:


t_scores = cross_val_score(GaussianNB(), messages_tfidf_vect.toarray(), y, cv = 10)


# In[117]:


t_scores.mean()


# In[126]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[127]:


boston.keys()


# In[128]:


data = boston['data']
target = boston['target']
feature_names = boston['feature_names']


# In[129]:


import numpy as np
import pandas as pd


# In[130]:


df = pd.DataFrame(data, columns = feature_names)
df['output'] = target


# In[131]:


df


# In[132]:


import matplotlib.pyplot as plt
import seaborn as sns 


# In[133]:


sns.pairplot(df)


# In[134]:


for i in feature_names:
    plt.scatter(df[i], df['output'])
    plt.xlabel(i)
    plt.show()


# In[137]:


df.iloc[:, :-1].corr()


# In[141]:


plt.figure(figsize = (20,20))
sns.heatmap(df.iloc[:, :-1].corr(),annot=True)


# In[142]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data, target)


# In[143]:


y_pred = model.predict(data)


# In[145]:


import numpy as np
residual = np.abs(y_pred-target)


# In[146]:


sns.distplot(residual, kde = True)


# In[147]:


from scipy.stats import probplot
fix, ax = plt.subplots(figsize = (10,5))
probplot(residual, plot = ax, fit = True)
plt.show()


# In[148]:


plt.scatter(target,residual)

