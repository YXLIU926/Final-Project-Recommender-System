import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import cufflinks
from plotly.offline import iplot
import json

nltk.download('stopwords')
# GLOBAL VAIRABLES THAT WILL BE USED IN TEXT CLEAN FUNCTIOn
stop_words = ""
sub_replace = None

def dataInfo(df):
    # HOTEL DESCRIPTION TOTAL WORDS
    vec = CountVectorizer().fit(df['desc'])
    bag_of_words = vec.transform(df['desc'])
    bag_of_words.toarray()
    print(bag_of_words.shape)
    # HOTEL DESCRIPTION TOTAL WORDS
    sum_words = bag_of_words.sum(axis=0)
    print(sum_words)

    # DESCRIPTION FREQUNCIES
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    print(words_freq)
    # DESCRIPTION FREQUNCIES IN ORDER
    words_freq = sorted(words_freq,key=lambda x:x[1],reverse=True)
    print(words_freq)

# TOP FREQUNCY WORDS
def get_top_n_words(corpus,n=None):
    vec = CountVectorizer(stop_words='english',ngram_range=(1,3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,key=lambda x:x[1],reverse=True)
    return words_freq[:n]

def clean_txt(text):
    text.lower()
    text = sub_replace.sub('',text)
    ' '.join(word for word in text.split() if word not in stop_words)
    return text

def recommendations(name,cosine_similarity, indices, df):
    recommended_hotels = []
    idx = indices[indices == name].index[0]
    print(idx)
    score_series = pd.Series(cosine_similarity[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series[1:1000].index)
    count = 0
    for i in top_10_indexes:
        hotelName = list(df.index)[i]
        url = list(df.url)[i]
        phone = list(df.phone)[i]
        street = list(df.street_address)[i]
        region = list(df.region)[i]
        city = list(df.city)[i]
        postal_code = list(df.postal_code)[i]
        hotelInfo = [hotelName, url, phone, street, region, city, postal_code]
        # recommended_hotels.append(list(df.index)[i])
        recommended_hotels.append(hotelInfo)
        count += 1
        if count == 10:
            break
    return recommended_hotels

def test():
    global sub_replace, stop_words
    # print(1)
    # print(stopwords)
    # LOAD DATA
    print("Loading Data")
    df = pd.read_csv(r'C:\Users\elton\Desktop\Kate_Project\code\clean_data.csv')
    # RPINT SOME DATA INFORMATION
    # dataInfo(df)
    # GET MOST COMMON WORDS FROM DESCRIPTION
    # common_words=get_top_n_words(df['desc'],20)
    # df3 = pd.DataFrame(common_words,columns=['desc','count'])
    # df3.groupby('desc').sum()['count'].sort_values().iplot(kind='barh',yTitle='Count',linecolor='black',title='top 20 before remove stopwords-ngram_range=(2,2)')
    print("Calculating Word Total For Each Review")
    df['word_count']=df['text'].apply(lambda x:len(str(x).split()))
    # print(df.head())
    # df1.groupby('desc').sum()['count'].sort_values().iplot(kind='barh',yTitle='Count',linecolor='black',title='top 20 before remove stopwords')

    sub_replace = re.compile('[^0-9a-z #+_]')
    stop_words = set(stopwords.words('english'))
    print("Removing All Stop Words")
    df['desc_clean'] = df['text'].apply(clean_txt)
    # print(df.head())
    
    df.set_index('name',inplace = True)
    tf=TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english')
    tfidf_matrix=tf.fit_transform(df['desc_clean'])
    print(tfidf_matrix.shape)
    print("Calculating Cosine Similarity For All Hotels (Item-Item Based)")
    cosine_similarity =linear_kernel(tfidf_matrix,tfidf_matrix)
    print(cosine_similarity.shape)
    # print(cosine_similarity[0])
    indices = pd.Series(df.index)
    print(indices[:5])
    print("Now Recommending")
    # return cosine_similarity
    print(recommendations('The Michelangelo Hotel',cosine_similarity, indices, df))

def readHotels():
    # dd = pd.read_json('review.txt')
    contents = []
    with open(r"C:\Users\elton\Desktop\Kate_Project\code\hotels.txt") as f:
        for line in f:
            contents.append(json.loads(line))
    format_rows = []
    for row in contents:
        # print(row)
        # break
        city = row["address"]["locality"]
        # if (city != "New York City"):
        #     continue
        id = row["id"]
        name = row["name"]
        if ("hotel_class" in row):
            hotel_class = row["hotel_class"]
        else:
            hotel_class = 0.0
        url = row["url"]
        phone = row["phone"]
        region = row["address"]["region"]
        if "street-address" in row["address"]:
            street_address = row["address"]["street-address"]
        else:
            street_address = ""
        if "postal-code" in row["address"]:
            postal_code = row["address"]["postal-code"]
        else:
            postal_code = 00000
        
        format_rows.append([id, name, hotel_class, url, phone, region, street_address, postal_code, city])
    data = pd.DataFrame(format_rows)
    data.columns = ['id', 'name', 'hotel_class', 'url', 'phone', 'region', 'street_address', 'postal_code', 'city']
    return data

def readReviews():
    # dd = pd.read_json('review.txt')
    contents = []
    dict = {}
    with open(r"C:\Users\elton\Desktop\Kate_Project\code\review.txt") as f:
        for line in f:
            contents.append(json.loads(line))
    format_rows = []
    for row in contents:
        id = row["offering_id"]
        if "service" in  row["ratings"]:
            service = row["ratings"]["service"]
        else:
            service = 0.0
        if "cleanliness" in  row["ratings"]:
            cleanliness = row["ratings"]["cleanliness"]
        else:
            cleanliness = 0.0
        if "location" in  row["ratings"]:
            location = row["ratings"]["location"]
        else:
            location = 0.0
        if "sleep_quality" in  row["ratings"]:
            sleep_quality = row["ratings"]["sleep_quality"]
        else:
            sleep_quality = 0.0
        if "rooms" in  row["ratings"]:
            rooms = row["ratings"]["rooms"]
        else:
            rooms = 0.0
        text = row["text"]
        if (id in dict):
            dict[id].append([id, service, cleanliness, location, sleep_quality, rooms, text])
        else:
            dict[id] = [[id, service, cleanliness, location, sleep_quality, rooms, text]]
    # for k,v in dict.items():
    #     print(k)
    #     print(v)
    #     break   
    for k,v in dict.items():
        id = k
        text = ""
        service = 0.0
        cleanliness = 0.0
        location = 0.0
        sleep_quality = 0.0
        rooms = 0.0
        count = 0
        for innerList in v:
            service += float(innerList[1])
            cleanliness += float(innerList[2])
            location += float(innerList[3])
            sleep_quality += float(innerList[4])
            rooms += float(innerList[5])
            text += innerList[6]
            count += 1
        service /= count
        cleanliness /= count
        location /= count
        sleep_quality /= count
        rooms /= count
        format_rows.append([id, service, cleanliness, location, sleep_quality, rooms, text])
    data = pd.DataFrame(format_rows)
    data.columns = ['id', 'service', 'cleanliness', 'location', 'sleep_quality', 'rooms', 'text']
    return data

def generateCleanDataCSV():
    hotel_df = readHotels()
    print(hotel_df.head())
    review_df = readReviews()
    print(review_df.head())
    print(len(review_df))
    final_df = pd.merge(hotel_df, review_df, on="id")
    final_df.to_csv(r'C:\Users\elton\Desktop\Kate_Project\code\clean_data.csv', index = False)
    print(len(final_df))

# generateCleanDataCSV()
test()

