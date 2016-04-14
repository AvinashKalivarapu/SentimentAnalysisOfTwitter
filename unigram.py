import re
import sys
import numpy as np
from sklearn import cross_validation
from sklearn.multiclass import OneVsOneClassifier
from svm_classifier import train_classifier #,predict_sentiment
from sklearn import svm
sys.path.insert(0,'ark-tokenizer')
from ark import tokenizeRawTweetText


def processTweet(tweet):
    tweet=tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','||URL||',tweet)
    tweet = re.sub('@[^\s]+','||AT_USER||',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    tweet=tweet.strip(' ')
    return tweet

fp = open("outtok", 'r')


#Create dictionary of words

wordict={}

pos_count=0
neg_count=0
neu_count=0

line=fp.readline()
while line:
    line=line.rstrip()
    fields=re.split(r'\t+',line)

    if "positive" == fields[0]:
        pos_count+=1
    elif "negative" == fields[0]:
        neg_count+=1
    else:
        neu_count+=1
    
    
    tokens=re.split(' ',fields[1])
    size=len(tokens)
    for i in range(size):
        wordict[tokens[i]]=0
    line=fp.readline()

wordlist=sorted(wordict)
wordcount=0
for word in wordlist:
    wordict[word]=wordcount;
    wordcount+=1

fp.close()

print pos_count+neg_count+neu_count

#create boolean matrix (no. of tweets)*(no. of words in dict)


pos_matrix = [[0 for i in range(wordcount)] for j in range(pos_count)]
neg_matrix = [[0 for i in range(wordcount)] for j in range(neg_count)]
neu_matrix = [[0 for i in range(wordcount)] for j in range(neu_count)]

def parse_to_classifier(pos_matrix,neg_matrix,neu_matrix):
	final_matrix = []
	map(lambda x: final_matrix.append(x+[0,]),pos_matrix)
	map(lambda x: final_matrix.append(x+[1,]),neg_matrix)
	map(lambda x: final_matrix.append(x+[2,]),neu_matrix)
	final_matrix = np.array(final_matrix)
	np.random.shuffle(final_matrix)
	part = len(final_matrix)/10
	train_X = final_matrix[:9*part,:-1]
	train_Y = final_matrix[:9*part,-1]
	test_X = final_matrix[9*part:,:-1]
	test_Y = final_matrix[9*part:,-1]
	print train_X,train_Y,test_X,test_X
	return train_X,train_Y,test_X,test_X
	


fp = open("outtok", 'r')
line=fp.readline()

pos=0
neg=0
neu=0

while line:
    line=line.rstrip()
    fields=re.split(r'\t+',line)
    tokens=re.split(' ',fields[1])
    
    size=len(tokens)
    
    if "positive"==fields[0]:
        for i in range(size):
            pos_matrix[pos][wordict[tokens[i]]]=1
        pos+=1

    elif "negative"==fields[0]:
        for i in range(size):
            neg_matrix[neg][wordict[tokens[i]]]=1
        neg+=1

    else:
        for i in range(size):
            neu_matrix[neu][wordict[tokens[i]]]=1
        neu+=1
        
    
    line=fp.readline()
train_X,train_Y,test_X,test_X = parse_to_classifier(pos_matrix,neg_matrix,neu_matrix)
print "dimension", len(train_X[0])
#trained_clf = train_classifier(LinearSVC(random_state=0),train_X,train_Y)
score = cross_validation.cross_val_score(OneVsOneClassifier(svm.SVC(kernel='sigmoid')),train_X,train_Y,cv=3)
print "average accuracy of svm ",score.mean() 



total_tweets=pos_count+neg_count+neu_count

#classifying
    
pos_prob=float(pos_count)/float(total_tweets)
neg_prob=float(neg_count)/float(total_tweets)
neu_prob=float(neu_count)/float(total_tweets)




while True:
    test_tweet=raw_input("Enter Tweet :")
    test_tweet=processTweet(test_tweet)
    ark_tokenised=tokenizeRawTweetText(test_tweet)

    tweet_size=len(ark_tokenised)

    pos_tfreq=[1 for i in range(tweet_size)]
    neg_tfreq=[1 for i in range(tweet_size)]
    neu_tfreq=[1 for i in range(tweet_size)]


    for i in range(pos_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and pos_matrix[i][wordict[ark_tokenised[j]]]==1:
                pos_tfreq[j]+=1

    print pos_tfreq
    for i in range(neg_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and neg_matrix[i][wordict[ark_tokenised[j]]]==1:
                neg_tfreq[j]+=1

    print neg_tfreq
    
    for i in range(neu_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and neu_matrix[i][wordict[ark_tokenised[j]]]==1:
                neu_tfreq[j]+=1

    print neu_tfreq
    
    pos_uni_prob=1
    for i in range(tweet_size):
        pos_uni_prob*=float(pos_tfreq[j])/float(pos_count+1)


    neg_uni_prob=1
    for i in range(tweet_size):
        neg_uni_prob*=float(neg_tfreq[j])/float(neg_count+1)

    neu_uni_prob=1
    for i in range(tweet_size):
        neu_uni_prob*=float(neu_tfreq[j])/float(neu_count+1)



    pos_given_tweet=pos_prob*pos_uni_prob

    neg_given_tweet=neg_prob*neg_uni_prob

    neu_given_tweet=neu_prob*neu_uni_prob


    print pos_given_tweet,neg_given_tweet,neu_given_tweet
    
    if pos_given_tweet>=neg_given_tweet:
        if pos_given_tweet>=neu_given_tweet:
            print "positive"
        else:
            print "neutral"
    else:
        if neg_given_tweet>=neu_given_tweet:
            print "negative"
        else:
            print "neutral"
    

