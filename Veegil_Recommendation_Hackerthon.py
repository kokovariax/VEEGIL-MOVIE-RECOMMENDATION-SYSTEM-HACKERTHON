#!/usr/bin/env python
# coding: utf-8

# AUTHOUR: Ukachukwu Christian Chinweike<br>
# 
# TITLE: Movie recommendation system<br>
# 
# LANGUAGE: Python

# INSTALL THE DEPENDENCIES REQUIRED FOR A GOOD CODING PRACTICE

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from ast import literal_eval


# Reading and Storing Data

# In[2]:


cre = pd.read_csv('tmdb_5000_credits.csv')
mov = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


#let us see what the credit data looks like.
cre.head()


# In[4]:


#let us check out how our Movie data look like as well
mov.head()


# In[5]:


#LET US JOIN TH TWO DATA SETS TO HELP US PULL OUT THE CAST FROM THE cre DATASET
# joining two datasets
#please notice the double t in tittle, it is so to prevent it from generating x and y in the system and disturbibg the dataframe
cre.columns=['id','tittle','cast','crew']
mov=mov.merge(cre,on='id')


# FROM THE DATA PROVIDED ABOVE, EXPECIALLY IN THE mov DATA FRAME, RATING AND VOTING PLACES A MOVIE ON HIGH PEDESTALS, HENCE WE CONSIDER THE TWO DATA FIELDS AS PARAMETER FOR FILTERING

# In[6]:


mean_mov= mov['vote_average'].mean()
mean_mov


# In[7]:


uq_mov= mov['vote_count'].quantile(0.9)
uq_mov


# FROM THE ABOVE, YOU WILL NOTICE THAT GIVEN AMOUNT OF MOVIES MADE IT ABOVE THE NORMAL RATING. IN FACT, THEY ARE IN THE TOP 10% OF THE MOVIES VOTED BY AUDIENCE
# 
# 
# NOW WE FILTER THE MOVIE DATASET IN SUCH A WAY THAT THE MOVIES IN THE TOP VOTE COUNTS QUALIFIED FOR THE STUDY; BESIDES WHO WOULDN'T WANT TO SEE INTERESTING MOVIES?

# In[8]:


qual=mov.copy().loc[mov['vote_count']>=uq_mov]
qual.shape


# In[9]:


#let us take a peep at the qualified movies
qual.head()


# 

# IMDb raton is a world class accepted rating for movies. The rating however is based on the average rating (1-10) and the vote count for any particular movie in question
# 
# The fomular is in form
# 
# (((vote count)/(vote count+ quantile(0.9) of vote count)* Average vote per movie) + ((vote count+ quantile(0.9) of vote count)/(vote count+ quantile(0.9) of vote count + vote count)* average rating)
# 
# from our Dataset we can give the formular as
# 
# **(v/(v+uq_mov)* R) + (uq_mov/(uq_mov+v)* mean_mov)

# 

# In[10]:


def weighted_rating(x,m=uq_mov,C=mean_mov):
    v=x['vote_count']
    R=x['vote_average']
    #calculation based on IMDB formula
    
    return (v /(v+m)* R) + (m/(m+v)*C)


# Each movie has a special IMDb weighted rating from the formular implemented now, hence we create another field that will describe the scores of a movie and call it "Score"

# In[11]:


qual['score']= qual.apply(weighted_rating,axis=1)


# Now let us pull out the top 20 rated movies based on their IMD rating scores among the qualified movies

# In[12]:


qual = qual.sort_values('score', ascending=False)

qual[['title', 'vote_count', 'vote_average', 'score']].head(20)


# HAVING GOTTEN THE FIRST PHASE, AT LEAST WE CAN RECOMMEND BASED ON RATING. BUT ALL THESE 481 MOVIES WON'T BE SEEN AS THE TRENDING U POPULAR MOVIES. HENCE, WE SORT THE MOVIE DATA FRAME BASED ON POPULARITY. WORKING WITH TOP 7 MOVIES

# In[13]:


pop= mov.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,5))

plt.barh(pop['title'].head(7),pop['popularity'].head(7), align='center',
        color='orange')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()


# FROM THE ABOVE, WE CAN PLACE THE TOP SEVEN MOST POPULAR MOVIES NOW ON THE TOP OF THE SCREEN TO ENABLE THE USERS CHECK THEM OUT.

# 

# THE NEXT FILTERING IS BASED ON RELATIONSHIP BETWEEN THE CONTENT OF THE MOVIE. HERE, WE WILL DO A LITTLE OF NATURAL LANGUAGE PROCESSING.

# In[14]:


#LETS SEE HOW THE MOVIE OVERVIEW IN MOVIE DATASET LOOKS LIKE
mov['overview'].head(5)


# In[15]:


#installing and configuring quick use dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(stop_words='english')


# 

# In[16]:


# Replace Nan with an empty string
mov['overview']=mov['overview'].fillna('')

# construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix= tfidf.fit_transform(mov['overview'])

tfidf_matrix.shape


# 

# In[17]:


from sklearn.metrics.pairwise import linear_kernel

#compute the cosine similarity matrix
cosine_sim= linear_kernel(tfidf_matrix,tfidf_matrix)


#  There is a need to use the movie title to control the recommendations no matter the parameters in check. so, we get a function taking movie title as an input and outputs a list of the 7 most similar movies. Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices. we should develop a pattern that will respond to the movies and pull out the movie's index in the source/master data according the imputs

# In[18]:


# construct a reverse map of indices and movie titles
indices= pd.Series(mov.index, index=mov['title']).drop_duplicates()


# In[19]:


# Function that takes in movie title & outputs similar movies
def rec_movie(title,cosine_sim=cosine_sim):
    # get the index of the movie that matches the title
    idx= indices[title]
    
    # get the pairwise similarity scores of all the movies with that movie
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #sort the movies based on similarity scores
    sim_scores= sorted(sim_scores,key=lambda x :x[1], reverse=True)
    
    # get the scors of the 7 most similar movies
    sim_scores=sim_scores[1:8]
    
    #get the movie indices
    movie_indices=[i[0] for i in sim_scores]
    
    # return the top 7 most similar movies
    return mov['title'].iloc[movie_indices]


# LET US TEST THE ABOVE MODEL AND SEE HOW IT WORKS
# 

# In[20]:


rec_movie('Inception')


# TA DAAA! ANOTHER ONE!

# In[21]:


rec_movie('The Shawshank Redemption')


# 

# THERE WILL BE OTHER PARAMETERS FOR RECOMMENDING A MOVIE APART FROM RATING, VOTES AND CONTENTS. I WOULD LOVE TO WATCH EVERY MOVIE BY VIN DIESEL AND WILL SMITH NO MATTER HOW MUCH GENESIS CINEMA OR FILMHOUSE CHARGES FOR IT. HENCE, WE PREPARE A RECOMMENDATION MODEL BASED ON THE ACTORS AND CREW, EVEN THE GANRE(ACTION, COMEDY AND EPIC)

# In[22]:


#HAVING MERGED IT, WE CAN NOW EXPLORE THE FILES EASILY
features=['cast','crew','keywords','genres']
for feature in features:
    mov[feature]= mov[feature].apply(literal_eval)


# LET US SEE HOW THE MOVIE DIRECTOR MATTER

# In[23]:


def mov_dir(x):
    for i in x:
        if i['job']=='Director':
            return i['name']
        return np.nan


# In[24]:


# return the list top 3 elements or entire list ; whichever is more.
def get_list(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        # check if more than 3 elements exists, If yes, return only first three. If no , return entire list
        if len(names)>3:
            names=names[:3]
            return names
        #return empty list in case of missing/malformed data
        return []


# FROM THE TOP THREE KEYWORD, GET THE DIRECTOR PER MOVIE, AND PROVIDE "NAN", THAT IS, NULL WHERE THERE IS NO DIRECTOR
# 

# In[25]:


mov['director']=mov['crew'].apply(mov_dir)
features=['cast','keywords','genres']
for feature in features:
    mov[feature]=mov[feature].apply(get_list)


# LET US SEE THE NEW DATA FRAME CONTAINING THE NEW FEATURES INTRODUCED

# In[26]:


mov[['title','cast','director','keywords','genres']].head(3)


# 

# SINCE WE ARE USING KEYWORDS, LET US MAKE ALL ENTIRS AND ALL FUNCTION VALUES TO BE IN SMALL LETTERS(LOWER CASE) SO THAT WHEN THE USER SEARCHES WITH BOTH CAPITAL AND SMALL LETTERS, IT MIGHT PULL A MATCH. ALSO, LET US REMOVE PUNCTUATIONS AD SPACES TO ENHANCE PROPER MATCHING 

# In[27]:


def cleandata(x):
    if isinstance(x,list):
        return[str.lower(i.replace(" ","")) for i in x]
    else:
        #check if director exists. If not, return empty string
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            return ''


# THE CODE BELOW APPLIES THE CLEANING FUNCTION BELOW INTO THE NEW FEATURE OF MOV DATA FRAME

# In[28]:


features=['cast','keywords','director','genres']

for feature in features:
    mov[feature]=mov[feature].apply(cleandata)


# Let us create a "soup" which string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords)

# In[29]:


def makesoup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
mov['soup'] = mov.apply(makesoup, axis=1)

HERE, WE COUNT THE PRESENCE OF A "CAST" OR "CREW" AS A CRITERIA FOR THEIR RATING. 
# In[30]:


from sklearn.feature_extraction.text import CountVectorizer

count= CountVectorizer(stop_words='english')
count_matrix= count.fit_transform(mov['soup'])


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity


# In[32]:


cosine_sim2=cosine_similarity(count_matrix,count_matrix)


# In[33]:


mov= mov.reset_index()
indices= pd.Series(mov.index, index=mov['title'])


# In[34]:


rec_movie('Stolen',cosine_sim2)


# In[35]:


rec_movie('Takers',cosine_sim2)


# In[ ]:





# In[ ]:




