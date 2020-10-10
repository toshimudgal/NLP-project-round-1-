
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')



f= open('/content/drive/My Drive/python/nlp project/A Tale of Two Cities by Charles Dickens.txt', 'r')
text1=f.read()
k= open('/content/drive/My Drive/python/nlp project/Pride and Prejudice by Jane Austen.txt', 'r')
text2=k.read()



from google.colab import drive
drive.mount('/content/drive')



from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string



def text_cleaning(text):
  # to remove special chars
  text= text.lower()
  text= re.sub('\[.*?\]', '', text)
  text = re.sub("\\W"," ",text) 
  # to remove links
  text = re.sub('https?://\S+|www\.\S+', '', text)
  #to remove punctuations
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  text= re.sub("chapter ","",text)
  return text
  
  
  
text1=text_cleaning(text1)
text2=text_cleaning(text2)
text1
text2



twords = text1.split()

print('Number of words in book 1 :', len(twords))
twords2 = text2.split()



print('Number of words in book 2 :', len(twords2))



number_of_characters = len(text1)
print('Number of characters in Book 1 with spaces :', number_of_characters)

number_of_characters = len(text2)
print('Number of characters in Book 2 with spaces :', number_of_characters)

data = text1.replace(" ","")
#get the length of the data
number_of_characters = len(data)
print('Number of characters in Book 1 without spaces :', number_of_characters)
data = text2.replace(" ","")
number_of_characters = len(data)
print('Number of characters in Book 2 without spaces :', number_of_characters)




num_lines = 0
with open('/content/drive/My Drive/python/nlp project/A Tale of Two Cities by Charles Dickens.txt', 'r') as f:
    for line in f:
        num_lines += 1
print("Number of lines in book 1:")
print(num_lines)



num_lines = 0
with open('/content/drive/My Drive/python/nlp project/Pride and Prejudice by Jane Austen.txt' ,'r') as f:
    for line in f:
        num_lines += 1
print("Number of lines in book 2:")
print(num_lines)



token1=word_tokenize(text1)
token2=word_tokenize(text2)
token1
token2



from nltk.probability import FreqDist
fdist1 = FreqDist(token1)
fdist2 = FreqDist(token2)
fdist1
fdist2



import operator
sorted_d1 = dict( sorted(fdist1.items(), key=operator.itemgetter(1),reverse=True))
sorted_d2 = dict( sorted(fdist2.items(), key=operator.itemgetter(1),reverse=True))
print('Dictionary in descending order by value : ',sorted_d1)
print('Dictionary in descending order by value : ',sorted_d2)



N =20
# Using items() + list slicing 
# Get first K items in dictionary 
out1 = dict(list(sorted_d1.items())[0: N]) 
out2 = dict(list(sorted_d2.items())[0: N])
# printing result 
print("Dictionary limited by K is : " + str(out1)) 
print("Dictionary limited by K is : " + str(out2)) 
a_dictionary = out1
keys = a_dictionary.keys()
values = a_dictionary.values()



plt.bar(keys, values)
a_dictionary = out2
keys = a_dictionary.keys()
values = a_dictionary.values()



plt.bar(keys, values)
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(text1) 
  
  
  
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
 plt.show() 



wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(text2) 
  
  
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
 plt.show() 



nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(nltk.corpus.stopwords.words())



def stopwords_cleaned(sentence):
    res = []
    for word in sentence:
        if word not in stopwords:
            res.append(word)
    return res
    

token3=stopwords_cleaned(token1)
token4=stopwords_cleaned(token2)



from nltk.probability import FreqDist
fdist3 = FreqDist(token3)
fdist4 = FreqDist(token4)



sorted_d1 = dict( sorted(fdist3.items(), key=operator.itemgetter(1),reverse=True))
sorted_d2 = dict( sorted(fdist4.items(), key=operator.itemgetter(1),reverse=True))
print('Dictionary in descending order by value : ',sorted_d1)
print('Dictionary in descending order by value : ',sorted_d2)
comment_words = '' 
comment_words += " ".join(token3)+" "



wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',  
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
comment_words = '' 
comment_words += " ".join(token4)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',  
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



words1 = '' 
words1 += " ".join(token3)+" "
words2 = '' 
words2 += " ".join(token4)+" "

from collections import defaultdict
x = {}
def get_word_counts(text):
   

    for word in text.split():
          
        if(len(word) not in x):
        	x[len(word)]=1
        else:
        	x[len(word)]+=1
    
(get_word_counts(words1))
x
(get_word_counts(words2))
x




nltk.download('averaged_perceptron_tagger')
tagged1 = nltk.pos_tag(token3) 
print(tagged1) 
tagged2 = nltk.pos_tag(token4) 
print(tagged2) 

dit = {}
for a,b in tagged1:
	if(b not in dit):
		dit[b]=1
	else:
		dit[b]+=1
dit2 = {}
for a,b in tagged2:
	if(b not in dit2):
		dit2[b]=1
	else:
		dit2[b]+=1


sorted_d1 = dict( sorted(dit.items(), key=operator.itemgetter(1),reverse=True))
sorted_d2 = dict( sorted(dit2.items(), key=operator.itemgetter(1),reverse=True))
print('Dictionary in descending order by value : ',sorted_d1)
print('Dictionary in descending order by value : ',sorted_d2)

N =15
# Using items() + list slicing 
# Get first K items in dictionary 
out1 = dict(list(sorted_d1.items())[0: N]) 
out2 = dict(list(sorted_d2.items())[0: N])
# printing result 
print("Dictionary limited by K is : " + str(out1)) 
print("Dictionary limited by K is : " + str(out2)) 

N =15
# Using items() + list slicing 
# Get first K items in dictionary 
out1 = dict(list(sorted_d1.items())[0: N]) 
out2 = dict(list(sorted_d2.items())[0: N])
# printing result 
print("Dictionary limited by K is : " + str(out1)) 
print("Dictionary limited by K is : " + str(out2)) 


a_dictionary = out2
keys = a_dictionary.keys()
values = a_dictionary.values()
plt.bar(keys, values)
plt.xlabel('TAGS')
plt.ylabel('Count')
plt.show()

