"""
Class to make an inference using the trained model.
"""
from sklearn.externals import joblib
import sys
from sklearn import preprocessing
import numpy as np

class Inference:
    def __init__(self):
        self.regions = ["US", "Latin_America", "Europe", "Asia", "South_America"]
        self.sentiments = ["happy", "sad", "neutral"]
        self.dayTimes = ["morning", "afternoon", "night"]
        self.brands = ["Fanta", "Coke", "Coke_Zero", "Diet_Coke", "Smart_Water", "Minute_Made", "Sobe", "Mineral_Water",
                  "Sprite"]
    
        # Instantiate Label Encoders
        self.regions_le = preprocessing.LabelEncoder().fit(self.regions)
        self.sentiments_le = preprocessing.LabelEncoder().fit(self.sentiments)
        self.dayTimes_le = preprocessing.LabelEncoder().fit(self.dayTimes)
        self.brands_le = preprocessing.LabelEncoder().fit(self.brands)
        
        # Load classifier
        self.clf = joblib.load(filename='clf.pkl')
        
    def infer(self, region, brand, text_len, sentiment, has_photo, has_vid, vid_len, dayTime):
        
        print('\n\nPredicting performance...')
        region = list(self.regions_le.transform(region))[0]
        brand = list(self.brands_le.transform(brand))[0]
        sentiment = list(self.sentiments_le.transform(sentiment))[0]
        tod = list(self.dayTimes_le.transform(dayTime))[0]
        array = np.array([region, brand, text_len[0], sentiment, has_photo[0], has_vid[0], vid_len[0], tod]).reshape(1, -1)
        pred = self.clf.predict(array)
        print("The model predicts {} likes!".format(pred[0]), '\n\n')
        
        
if __name__ == '__main__':
    inf = Inference()
    region = np.array([sys.argv[1]])
    brand = np.array([sys.argv[2]])
    text_len = np.array([sys.argv[3]])
    sentiment = np.array([sys.argv[4]])
    has_photo = np.array([sys.argv[5]])
    has_vid = np.array([sys.argv[6]])
    vid_len = np.array([sys.argv[7]])
    tod = np.array([sys.argv[8]])
    inf.infer(region,
              brand, text_len, sentiment, has_photo, has_vid, vid_len, tod)
    
        