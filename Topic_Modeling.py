#!/usr/bin/env python
# coding: utf-8


# * **Importing libraries**

# In[3]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

# natural language toolkit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  
le=WordNetLemmatizer()

import re

from tqdm import tqdm
tqdm.pandas(desc="my bar!")

import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} textblob')
from textblob import TextBlob

import scipy.stats as stats
from collections import Counter
from sklearn.manifold import TSNE

get_ipython().system('{sys.executable} -m pip install wordcloud')
from wordcloud import WordCloud


# In[41]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


# In[42]:


from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

import sys
get_ipython().system('{sys.executable} -m pip install gensim')
import gensim
from gensim import models, corpora
from gensim.models import LsiModel
from gensim import similarities
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf

from collections import Counter
from operator import itemgetter


# In[53]:


get_ipython().system('{sys.executable} -m pip install pyLDAvis')
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

import matplotlib.pyplot as plt
import seaborn as sns


# * Importing the dataset

# In[54]:

#customizable location of the dataset files
documents = pd.read_csv('/Users/evagao/Desktop/modular_code/input/documents.csv')
len(documents)


# In[55]:


documents.head()


# * Splitting documents into train and test where train consists of 90% of the documents and test 10%.
# * The reason 10% for the test is just to give an idea how to find a topic/topic(s) for unseen documents and 90% for the train might have a sufficient data to train

# In[56]:


train_documents = documents[:int(len(documents)*0.9)]
test_documents  = documents[int(len(documents)*0.9):]
print(len(train_documents), len(test_documents)) 
assert len(train_documents)+len(test_documents) == len(documents)


# * **Cleaning the documents**

# * Replacing the short form having apostrophe with the complete form

# In[57]:


contractions = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}





dataset = [train_documents, test_documents]




def clean_documents(text):
    
    if pd.isnull(text):
        return text
    
    text = text.lower()
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
            
    text         = re.sub("[^a-zA-Z#]", ' ', text)
    word_tokens  = nltk.word_tokenize(text)
    tokens       = [le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
    cleaned_text =" ".join(tokens)
    
    return cleaned_text





train_documents['clean_document'] = train_documents['document'].progress_apply(lambda x: clean_documents(x))




train_documents.head()





test_documents['clean_document'] = test_documents['document'].progress_apply(lambda x: clean_documents(x))





train_docs = train_documents['clean_document'].copy()
test_docs  = test_documents['clean_document'].copy()





# Define helper functions
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their 
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for 
             word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])





# Define helper functions
def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)




count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=15,
                                     count_vectorizer=count_vectorizer, 
                                     text_data=train_documents['clean_document'])

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), word_values);
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
ax.set_title('Top words in headlines dataset (excluding stop words)');
ax.set_xlabel('Word');
ax.set_ylabel('Number of occurences');
plt.show()



def stats_of_documents(data_df):
    
    data = data_df['clean_document'].tolist()
    tagged_headlines = [TextBlob(data[i]).pos_tags for i in range(len(data))]
    
    tagged_headlines_df = pd.DataFrame({'tags':tagged_headlines})

    word_counts = [] 
    pos_counts = {}

    for headline in tagged_headlines_df[u'tags']:
        word_counts.append(len(headline))
        for tag in headline:
            if tag[1] in pos_counts:
                pos_counts[tag[1]] += 1
            else:
                pos_counts[tag[1]] = 1

    print('Total number of words: ', np.sum(word_counts))
    print('Mean number of words per headline: ', np.mean(word_counts))
    print('Minimum number of words per headline: ', np.min(word_counts))
    print('Maximum number of words per headline: ', np.max(word_counts))

    pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
    pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(18,8))
    ax.bar(range(len(pos_counts)), pos_sorted_counts);
    ax.set_xticks(range(len(pos_counts)));
    ax.set_xticklabels(pos_sorted_types);
    ax.set_title('Part-of-Speech Tagging for Headlines Corpus');
    ax.set_xlabel('Type of Word');


# In[70]:


nltk.download('averaged_perceptron_tagger')
stats_of_documents(train_documents)




lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top   = lsa_model.fit_transform(vect_text)





l=lsa_top[1]
print("Document 0 :")
for i,topic in enumerate(l):
    print(i,topic)


#

def topics_document(model_output, n_topics , data):
    '''
    returns a dataframe for each document having topic weightages
    and the dominant topic for each doc. 
    '''
    
    # column names
    topicnames = ["Topic" + str(i) for i in range(n_topics)]
    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic["dominant_topic"] = dominant_topic
    
    return df_document_topic




documet_topic_lsa = topics_document(lsa_top, n_topics=10, data=train_documents)
documet_topic_lsa



lsa_keys = get_keys(lsa_top)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)




topics_df = pd.DataFrame({'topic' : lsa_categories, 'count' : lsa_counts})
sns.barplot(x=topics_df['topic'], y = topics_df['count'])
plt.show()



vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")




# Define helper functions
def get_mean_topic_vectors(keys, two_dim_vectors, n_topics):
    '''
    returns a list of centroid vectors from each predicted topic category
    '''
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])    
        
        articles_in_that_topic = np.vstack(articles_in_that_topic)
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors





colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:10]





tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_top)





lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors, n_topics=10)

plot = figure(title="t-SNE Clustering of {} LSA Topics".format(10), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])

for t in range(10):
    label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
                   text_color=colormap[t])
    plot.add_layout(label)
    
show(plot)




# Generate a word cloud image for given topic
def draw_word_cloud(index, model):
    imp_words_topic=""
    comp=model.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]

    wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
    plt.figure( figsize=(5,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# In[126]:


draw_word_cloud(0, lsa_model)


# In[127]:


draw_word_cloud(1, lsa_model)


# In[128]:


draw_word_cloud(5, lsa_model)


# * **Predicting the topics on unseen documents**

# In[129]:


stats_of_documents(test_documents)


# * vect is the object of the TfidfVectorizer

# In[130]:


vect_text_test      = vect.transform(test_documents['clean_document'])
lsa_top_test        = lsa_model.transform(vect_text_test)


# In[131]:


lsa_keys_test = get_keys(lsa_top_test)
lsa_categories_test, lsa_counts_test = keys_to_counts(lsa_keys_test)

topics_df_test = pd.DataFrame({'topic' : lsa_categories_test, 'count' : lsa_counts_test})
sns.barplot(x=topics_df_test['topic'], y = topics_df_test['count'])
plt.show()





documet_topic_lsa_test = topics_document(lsa_top_test, n_topics=10, data=test_documents)
documet_topic_lsa_test




vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,# minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}')  # num chars > 3
data_vectorized = vectorizer.fit_transform(train_documents['clean_document'])


# * Training the model




# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes




# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))
# See model parameters
print(lda_model.get_params())





# Define Search Param
search_params = {'n_components': [10, 15, 20], 'learning_decay': [.5, .7, .9]}
# Init the Model
lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(data_vectorized)


# In[237]:


# Best Model
best_lda_model = model.best_estimator_
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


# * Transforming the vectorized data which is tranformed using CountVectorizer using the best lda model searched using GridSearchCV

# In[83]:


lda_output = best_lda_model.transform(data_vectorized)


# In[405]:


documet_topic_lda = topics_document(lda_output, n_topics=10, data=train_documents)
documet_topic_lda


# In[242]:


lda_keys = get_keys(lda_output)
lda_categories, lda_counts = keys_to_counts(lda_keys)

topics_df_lda = pd.DataFrame({'topic' : lda_categories, 'count' : lda_counts})
sns.barplot(x=topics_df_lda['topic'], y = topics_df_lda['count'])
plt.show()


# * Here the topics are distributed very well as compared to LSA and the major topic is topic 5

# In[244]:


# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=10):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# In[85]:


tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_output)


# In[86]:


lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors, n_topics=10)

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(10), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], color=colormap[lda_keys])

for t in range(10):
    label = Label(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 
                  text_color=colormap[t])
    plot.add_layout(label)

show(plot)


# * This is a much better result! Controlling for  t -SNE, it would seem that LDA has had much more succcess than LSA in separating out the topic categories. For this reason, LDA appears the more appropriate algorithm when we scale up the clustering process in the next section.

# In[87]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { max-width:100% !important; }</style>"))
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
display(HTML("<style>.output_area { max-width:100% !important; }</style>"))
display(HTML("<style>.input_area { max-width:100% !important; }</style>"))


# * **pyLDAvis** is a interactive LDA visualization python package. 
# *  The area of circle represents the importance of each topic over the entire corpus, the distance between the center of circles indicate the similarity between topics.
# * For each topic, the histogram on the right side listed the top 30 most relevant terms. LDA helped me extracted 10 main topics.

# In[88]:


#this is by far the easiest way to load a pre-trained sklearn lda model onto pyLDAvis
#there are other means of using this visible on pyLDAvis' main repo

display_data = pyLDAvis.sklearn.prepare(best_lda_model, #our pre-trained LDA model
                         data_vectorized, #this gives us our document-term matrix
                        vectorizer) #the vectoriser object
                        
pyLDAvis.display(display_data)  



def clean_documents_tokens(text):
    
    if pd.isnull(text):
        return text
    
    text = text.lower()
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
            
    text         = re.sub("[^a-zA-Z#]", ' ', text)
    word_tokens  = nltk.word_tokenize(text)
    tokens       = [le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
    
    return tokens


# In[90]:


train_documents['clean_tokens'] = train_documents['document'].progress_apply(lambda x: clean_documents_tokens(x))

texts = train_documents['clean_tokens']


# In[300]:


# Use Gensim's NMF to get the best num of topics via coherence score

# Create a dictionary
# In gensim a dictionary is a mapping between words and their integer id
dictionary = Dictionary(texts)

# Filter out extremes to limit the number of features
dictionary.filter_extremes(
    no_below=3,
    no_above=0.85,
    keep_n=4000
)

# Create the bag-of-words format (list of (token_id, token_count))
corpus = [dictionary.doc2bow(text) for text in texts]


# In[383]:


# Build LDA model
lda_model_ = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)





# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model_, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# * **Predictions on the unseen dataset**

# In[407]:


data_vectorized_test      =vectorizer.transform(test_documents['clean_document'])
lda_top_test              =best_lda_model.transform(data_vectorized_test)


# In[406]:


documet_topic_lda_test = topics_document(lda_top_test, n_topics=10, data=test_documents)
documet_topic_lda_test




tfidf_vectorizer = TfidfVectorizer(
    min_df=3,
    max_df=0.85,
    max_features=4000,
    ngram_range=(1, 2),
    preprocessor=' '.join
)
tfidf = tfidf_vectorizer.fit_transform(texts)


# In[310]:


# Create a list of the topic numbers we want to try
topic_nums = list(np.arange(7, 25, 3))


# In[311]:


# Run the nmf model and calculate the coherence score
# for each number of topics
coherence_scores = []

for num in topic_nums:
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )
    
    # Run the coherence model to get the score
    cm = CoherenceModel(
        model=nmf,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    coherence_scores.append(round(cm.get_coherence(), 5))

# Get the number of topics with the highest coherence score
scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]


# In[312]:


# Plot the results
fig = plt.figure(figsize=(15, 7))

plt.plot(
    topic_nums,
    coherence_scores,
    linewidth=3,
    color='#4287f5'
)

plt.xlabel("Topic Num", fontsize=14)
plt.ylabel("Coherence Score", fontsize=14)
plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), fontsize=18)
plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# * Fitting * **13** topics using NMF

# In[92]:


nmf = NMF(n_components=13, random_state=43, init='nndsvd')
nmf_output = nmf.fit_transform(tfidf)

nmf_feature_names = tfidf_vectorizer.get_feature_names()
nmf_weights = nmf.components_


# In[314]:


# get topics with their terms and weights
def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    return topics


# prints components of all the topics
# obtained from topic modeling
def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms]) if num_terms else topic
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms]) if num_terms else tw

# prints components of all the topics
# obtained from topic modeling
def get_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     num_terms=None):

    topic_terms = []

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        topic_terms.append(topic[:num_terms] if num_terms else topic)

    return topic_terms

def getTermsAndSizes(topic_display_list_item):
    terms = []
    sizes = []
    for term, size in topic_display_list_item:
        terms.append(term)
        sizes.append(size)
    return terms, sizes


# * Important terms in each Topic

# In[315]:


topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)
print_topics_udf(topics, total_topics=2, num_terms=30, display_weights=True)


# * Visualize NMF Topics

# In[317]:


topics_display_list = get_topics_udf(topics, total_topics=2, num_terms=30)


# In[318]:


terms, sizes = getTermsAndSizes(topics_display_list[0])

num_top_words = 30
fontsize_base = 30 / np.max(sizes) # font size for word with largest share in corpus

num_topics = 1

for t in range(num_topics):
    fig, ax = plt.subplots(1, num_topics, figsize=(6, 12))
    plt.ylim(0, num_top_words + 1.0)
    plt.xticks([])
    plt.yticks([])
    plt.title('Topic #{}'.format(t))

    for i, (word, share) in enumerate(zip(terms, sizes)):
        word = word + " (" + str(share) + ")"
        plt.text(0.3, num_top_words-i-1.0, word, fontsize=fontsize_base*share)

plt.tight_layout()





terms, sizes = getTermsAndSizes(topics_display_list[1])

num_top_words = 30
fontsize_base = 160 / (np.max(sizes))*0.8 # font size for word with largest share in corpus

num_topics = 1

for t in range(num_topics):
    fig, ax = plt.subplots(1, num_topics, figsize=(16, 30))
    plt.ylim(0, num_top_words + 1.0)
    plt.xticks([])
    plt.yticks([])
    plt.title('Topic #{}'.format(t+1))

    for i, (word, share) in enumerate(zip(terms, sizes)):
        word = word + " (" + str(share) + ")"
        plt.text(0.3, num_top_words-i-.5, word, fontsize=fontsize_base*share)

plt.tight_layout()




documet_topic_nmf = topics_document(nmf_output, n_topics=13, data=train_documents)
documet_topic_nmf




nmf_keys = get_keys(nmf_output)
nmf_categories, nmf_counts = keys_to_counts(nmf_keys)

topics_df_nmf = pd.DataFrame({'topic' : nmf_categories, 'count' : nmf_counts})
sns.barplot(x=topics_df_nmf['topic'], y = topics_df_nmf['count'])
plt.show()





vocab = tfidf_vectorizer.get_feature_names()

for i, comp in enumerate(nmf.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")





panel = pyLDAvis.sklearn.prepare(nmf, tfidf, tfidf_vectorizer, mds='tsne')
pyLDAvis.display(panel)


# * **Predictions on the unseen dataset**



test_documents['clean_tokens'] = test_documents['document'].progress_apply(lambda x: clean_documents_tokens(x))





data_vectorized_test      =tfidf_vectorizer.transform(test_documents['clean_tokens'])
nmf_top_test              =nmf.transform(data_vectorized_test)





documet_topic_nmf_test = topics_document(nmf_top_test, n_topics=13, data=test_documents)
documet_topic_nmf_test





