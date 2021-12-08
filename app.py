import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import streamlit as st
nltk.download('stopwords')

# GLOBAL VAIRABLES THAT WILL BE USED IN TEXT CLEAN FUNCTION & MODEL BUILD
stop_words = ""
sub_replace = None

def clean_txt(text):
    text = str(text)
    text.lower()
    text = sub_replace.sub('',text)
    ' '.join(word for word in text.split() if word not in stop_words)
    return text

#########################################################################

# TOP FREQUNCY WORDS
def get_top_n_words(corpus,n=None):
    vec = CountVectorizer(stop_words='english',ngram_range=(1,3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,key=lambda x:x[1],reverse=True)
    return words_freq[:n]


def recommendations(name, hotel_class_required, cleanliness_required, sleep_quality_required, cosine_similarity, indices, df):
    recommended_hotels = []
    idx = indices[indices == name].index[0]
    score_series = pd.Series(cosine_similarity[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series[1:1000].index)
    count = 0
    hotel_class_boolean = False
    tempList = []
    if len(hotel_class_required) > 0:
        hotel_class_boolean = True
        for hotelClass in hotel_class_required:
            tempList.append(int(hotelClass))
        hotel_class_required = tempList
    for i in top_10_indexes:
        hotelName = list(df.index)[i]
        url = list(df.url)[i]
        phone = list(df.phone)[i]
        street = list(df.street_address)[i]
        region = list(df.region)[i]
        city = list(df.city)[i]
        if (city != "New York City"):
            continue
        postal_code = list(df.postal_code)[i]
        hotel_class = int(list(df.hotel_class)[i])
        if hotel_class_boolean and hotel_class not in hotel_class_required:
            continue
        cleanliness = float(list(df.cleanliness)[i])
        # USER SET CLEANLINESS REQUIREMENT
        if (cleanliness_required[0] != cleanliness_required[1]):
            lowerBound = cleanliness_required[0]
            higherBound = cleanliness_required[1]
            if cleanliness < lowerBound or cleanliness > higherBound:
                continue
        sleep_quality = float(list(df.sleep_quality)[i]) 
        # UsER SET SLEEP QUALITY REQUIREMENT
        if (sleep_quality_required[0] != sleep_quality_required[1]):
            lowerBound = sleep_quality_required[0]
            higherBound = sleep_quality_required[1]
            if sleep_quality < lowerBound or sleep_quality > higherBound:
                continue
        hotelInfo = [hotelName, url, phone, street, region, city, postal_code, str(hotel_class), str(cleanliness), str(sleep_quality)]
        # recommended_hotels.append(list(df.index)[i])
        recommended_hotels.append(hotelInfo)
        count += 1
        if count == 10:
            break
    return recommended_hotels


@st.cache(allow_output_mutation=True)
def load_model():
    global stop_words
    global sub_replace
    seattleHotels = []
    #generateCleanDataCSV()
    sub_replace = re.compile('[^0-9a-z #+_]')
    stop_words = set(stopwords.words('english'))
    print("Loading Data")
    path = "C:\\Users\\yunxinliu\\Documents\\GitHub"
    my_file = path+'\\clean_data.csv'
    df = pd.read_csv(my_file)
    #df = pd.read_csv("test.csv")
    print(df.head())
    tempdf = df[['name', 'city']]
    for index, row in tempdf.iterrows():
        if row['city'] == "Seattle":
            seattleHotels.append(row['name'])
    seattleHotels.sort()
    print("Calculating Word Total For Each Review")
    df['word_count']=df['text'].apply(lambda x:len(str(x).split()))

    print("Removing All Stop Words")
    df['desc_clean'] = df['text'].apply(clean_txt)
    # df = pd.read_csv(r'C:\Users\yunxinliu\Documents\GitHub\cleaned_text.csv')
    # print(df.head())
    df.set_index('name',inplace = True)
    print(df.head())
    tf=TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english')
    tfidf_matrix=tf.fit_transform(df['desc_clean'])
    print(tfidf_matrix.shape)
    print("Calculating Cosine Similarity For All Hotels (Item-Item Based)")
    cosine_similarity =linear_kernel(tfidf_matrix,tfidf_matrix)
    print(cosine_similarity.shape)
    # print(cosine_similarity[0])
    indices = pd.Series(df.index)
    print("Finished ")
    return cosine_similarity, indices, df, seattleHotels


def turn_class_into_star(hotelInfo):
    hotelclass=int(hotelInfo)
    if hotelclass == 1:
        st.write("**_Hotel Class_**: ★☆☆☆☆")
    if hotelclass == 2:
        st.write("**_Hotel Class_**: ★★☆☆☆")
    if hotelclass == 3:
        st.write("**_Hotel Class_**: ★★★☆☆")
    if hotelclass == 4:
        st.write("**_Hotel Class_**: ★★★★☆")
    if hotelclass == 5:
        st.write("**_Hotel Class_**: ★★★★★")
        
def turn_cleanliness_into_mean(hotelInfo):
    cleanliness = float(hotelInfo)
    st.write("**_Cleanliness Rating_**:", "{:.2f}".format(cleanliness))
    
def turn_sleep_into_mean(hotelInfo):
    sleepquality = float(hotelInfo)
    st.write("**_Sleep Comfort Rating_**:", "{:.2f}".format(sleepquality))
        
if __name__ == '__main__':
    # LOAD MODEL
    cosine_similarity, indices, df, seattleHotels= load_model()
    ##############################Streamlit Appp###################################
    st.title("Seattle & New York Similar Hotel Recommender App")
    st.subheader('Overview')
    st.markdown("Author: Yunxin Liu")
    st.markdown("A hotel recommendation system that will make use of cosine similarity to output similar recommendations in New York based on selected features from a Seattle's hotel.")

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#989797;" /> """, unsafe_allow_html=True)
    st.markdown("Hit the **_Start my search_**🔎 button on the sidebar to pick your favorite hotels in New York!")
    st.text("______________________________________________________________________________________________________________________________________________________________________________________")

    with st.sidebar.form(key="Form1"):
        with st.sidebar:
            st.sidebar.markdown("Enter a hotel name from **_Seattle_** that you enjoyed your stay, and hit the search button. This app will fetch up to 10 similar ones in **_New York_** for you")
            # Host_Country = st.selectbox('Select HomeTeamName name:',('France', 'Spain', 'Italy', 'England', 'Belgium', 'Portugal','Sweden'))
            user_input=st.sidebar.selectbox('',tuple(seattleHotels))
            hotelName = str(user_input)
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.markdown('**_Optional_**: Filter the hotels by ratings')
            five_star = st.sidebar.checkbox('★★★★★')    
            four_star = st.sidebar.checkbox('★★★★☆')
            three_star = st.sidebar.checkbox('★★★☆☆')
            two_star = st.sidebar.checkbox('★★☆☆☆')
            one_star = st.sidebar.checkbox('★☆☆☆☆')    
            hotel_classes = []
            if (five_star):
                hotel_classes.append(5)
            if (four_star): 
                hotel_classes.append(4)
            if (three_star):
                hotel_classes.append(3)
            if (two_star):
                hotel_classes.append(2)
            if (one_star):
                hotel_classes.append(1) 
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.markdown('**_Optional_**: Preferred range of cleanliness (Select a range; Otherwise the default will be from 1 through 5)')
            clean=st.sidebar.slider("Cleanliness level",value=(1,1),max_value=5)
            st.sidebar.text("")
            st.sidebar.text("")
            st.sidebar.markdown('**_Optional_**: Preferred range of sleep comfort level (Select a range; Otherwise the default will be from 1 through 5)')
            sleep=st.sidebar.slider("Sleep quality",value=(1,1),max_value=5)
            st.sidebar.text("")
            st.sidebar.text("")       
            submitted1 = st.form_submit_button(label = 'Start my search!🔎')
    if (submitted1):
        if len(hotelName) == 0:
            print("Hotel Name Not Entered")
            st.write("Please Enter Hotel Name in Seattle")
        else:
            st.write("\n**_Top 10 Similar Hotels in New York based on your selections_**:")
            # print(hotelName)
            # print(indices.head())
            #print(clean)
            hotelList = recommendations(hotelName, hotel_classes, clean, sleep, cosine_similarity, indices, df)
            resultpd = pd.DataFrame(hotelList)
            #st.table(resultpd)
            i = 1
            for hotelInfo in hotelList:
                st.write("**_Reccommendation_**",i)  
                st.write("**_Hotel Name_**:", hotelInfo[0] , "\n")
                st.write(hotelInfo[1] , "\n")
                st.write("**_Address_**:",hotelInfo[3],"\n")
                turn_class_into_star(hotelInfo[7])
                turn_cleanliness_into_mean(hotelInfo[8])
                turn_sleep_into_mean(hotelInfo[9])
                st.text("______________________________________________________________________________________________________________________________________________________________________________________")
                i+=1
    st.sidebar.text("")
    st.sidebar.text("")   
    st.sidebar.text("")
    st.sidebar.text("") 
    st.markdown("The dataset came from https://www.cs.cmu.edu/~jiweil/html/hotel-review.html. The author crawled 878,561 reviews from 4,333 hotels in more than 10 states through TripAdvisor. This app used the data from New York and Seattle to speed up the runtime.")
