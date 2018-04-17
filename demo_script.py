import numpy as np
import random

if __name__ == "__main__":
    datFile = open("test_data.csv", "w+")

    datFile.write("Region,Text_Length,Sentiment,Has_Photo,Has_Video,Video_Length,Time_Of_Day,Likes\n")

    regions = ["US", "Latin America", "Europe", "Asia", "South America"]
    sentiments = ["happy", "sad", "neutral"]
    dayTimes = ["morning", "afternoon", "night"]

    for i in range(1000):
        reg = np.random.choice(regions, size=1)[0]
        txtLen = random.randint(25, 200)
        senti = np.random.choice(sentiments, size=1)[0]
        hasPhot = random.randint(0, 1)
        hasVid = random.randint(0, 1)
        if hasVid:
            vid_length = random.randint(10, 120)
        else:
            vid_length = "N/A"
        tod = np.random.choice(dayTimes, size=1)[0]
        likes = random.randint(300, 4000)
        ln = reg + "," + str(txtLen) + "," + senti + "," + str(hasPhot) + "," + str(hasVid) \
             + "," + str(vid_length) + "," + tod + "," + str(likes) + "\n"
        datFile.write(ln)
