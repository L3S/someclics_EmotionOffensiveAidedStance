The data contains tweet IDs and annotations in the below format:

tweetid;emoji;stance;emotion;toxic_multi

The emotion list labels belong to different emotions categories in order: ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',"positive","negative"]

Similarly offensive categories are in order: ['SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT','SEXUALLY_EXPLICIT','TOXICITY']

Since, we share our dataset only with the tweet ids and annotations for privacy issues, it is possible to use fetchTweetFromID.py python script to fetch the tweet objects with the tweet id.
