import numpy as np
import random
from sklearn import preprocessing
from sklearn.externals import joblib


if __name__ == "__main__":
    datFile = open("test_data.csv", "w+")

    datFile.write("Region,Brand, Text_Length,Sentiment,Has_Photo,Has_Video,Video_Length,Time_Of_Day,Likes\n")

    regions = ["US", "Latin_America", "Europe", "Asia", "South_America"]
    sentiments = ["happy", "sad", "neutral"]
    dayTimes = ["morning", "afternoon", "night"]
    brands = ["Fanta", "Coke", "Coke_Zero", "Diet_Coke", "Smart_Water", "Minute_Made", "Sobe", "Mineral_Water", "Sprite"]
    
    # Instantiate Label Encoders
    regions_le = preprocessing.LabelEncoder()
    sentiments_le = preprocessing.LabelEncoder()
    dayTimes_le = preprocessing.LabelEncoder()
    brands_le = preprocessing.LabelEncoder()
    tod_le = preprocessing.LabelEncoder()

    # Do the label encoding
    regions_le = regions_le.fit(regions)
    regions = regions_le.transform(regions)
    sentiments_le = sentiments_le.fit(sentiments)
    sentiments = sentiments_le.transform(sentiments)
    dayTimes_le = dayTimes_le.fit(dayTimes)
    dayTimes = dayTimes_le.transform(dayTimes)
    brands_le = brands_le.fit(brands)
    brands = brands_le.transform(brands)
    

    for i in range(1000):
        reg = np.random.choice(regions, size=1)[0]
        txtLen = random.randint(25, 200)
        senti = np.random.choice(sentiments, size=1)[0]
        brand = np.random.choice(brands, size=1)[0]
        hasPhot = random.randint(0, 1)
        hasVid = random.randint(0, 1)
        if hasVid:
            vid_length = random.randint(10, 120)
        else:
            vid_length = "N/A"
        tod = np.random.choice(dayTimes, size=1)[0]
        likes = random.randint(300, 4000)
        ln = str(reg) + "," + str(brand) + "," + str(txtLen) + "," + str(senti) + "," + str(hasPhot) + "," + str(hasVid) \
             + "," + str(vid_length) + "," + str(tod) + "," + str(likes) + "\n"
        datFile.write(ln)
