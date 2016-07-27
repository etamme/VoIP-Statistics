from random import randint
from random import shuffle
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

# a histogram that represents the values, and the percentage of the total they 
# make up in the generated data set
#
# the key is the percentage, and the value is a two element array with
# a minimum, and maximum value
#
# NOTE - the min/max values will be divided by 100
#
# EXAMPLE
# .10-.29 = 20%
# .50-.65 = 75%
# .90-5.00 = 5%
#
# callHist={'20':[10,290],
#           '75':[50,65],
#           '5' :[90,500]}

callHist={'19':[20,30],
          '30':[30,40],
          '21':[50,70],
          '20':[70,80],
          '10':[80,90]}
# total number of data points
numCalls=5000

# create fraudulent traffic?
fraud=1
# total number of fraudulent calls to create
numFraudCalls=500
fraudHist={'90':[100,300],
           '10':[400,500]}

# calculates euclidean distance between two lists
def hellinger(p, q):
  return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def generateData(hist,numcalls):
  print hist
  data=[]
  for k,v in hist.iteritems():
    minVal=int(min(v))
    maxVal=int(max(v))
    percent=float(k)
    calls=numcalls*(percent/100.0)
    print str(calls)+" between "+str(minVal)+" - "+str(maxVal)
    for c in range(int(calls)):
      val=randint(minVal,maxVal)/100.0
      data.append(val)
  shuffle(data)
  return data

# calculate helligers distance on a moving set of 20 calls
def generateDistance(data):
  distData=[]
  # the first 20 calls is used as a fixed training set to compare against
  # a moving window of 20 calls.  This set is normalized.
  dist1= [float(i)/sum(data[:20]) for i in data[:20]]
  dist2=list(data[:20])
  # calculate the distance on the moving window
  for v in data:
    dist2.pop(0)
    dist2.append(v)
    distData.append(hellinger(dist1,[float(i)/sum(dist2) for i in dist2]))
  return distData

# generate data sets
data=generateData(callHist,numCalls)

if fraud==1:
  # make a copy of the data before injecting fraud so we have a training set
  nofraud=list(data)
  fraudData=generateData(fraudHist,numFraudCalls)
  # slice and dice the data so that the fraud is inserted into the middle, and
  # mixed in with the non-fraudulent data
  mididx=numCalls/2
  front=data[0:mididx]
  end=data[mididx+1:]
  mididx=len(end)/2
  middle=end[0:mididx]+fraudData
  shuffle(middle)
  tail=end[mididx+1:]
  data=front+middle+tail

# create our list of distances
distData=generateDistance(data)

if fraud==1:
  # buckets for scaling price histogram 
  buckets = [0,1,2,4,9,16,25,36,49]
  # create a list of distance based on no fraud, so we can calculate
  # avg and stdev without fraud traffic included
  fraudDistData=generateDistance(nofraud)
  distAvg=sum(fraudDistData)/len(fraudDistData)
  distStd=np.std(fraudDistData)
else:
  # buckets for scaling price histogram 
  buckets = [0.01,0.10,0.20,0.40,0.90,1.6,2.5,3.6,4.9]
  distAvg=sum(distData)/len(distData)
  distStd=np.std(distData)

avg=sum(data)/len(data)
threshold=distAvg+(2*distStd)
hits=sum( i > threshold for i in distData)

hitx=0
hitcount=0
for v in distData:
  if v > threshold:
    hitcount+=1
  if hitcount >= 10:
    break
  hitx+=1
  if hitx % 100 == 0:
    hitcount=0

print "avg: "+str(avg)
print "distavg: "+str(distAvg)
print "diststd: "+str(distStd)
print "threshold: "+str(threshold)
print "hits: "+str(hits)

plt.hist(data,buckets, histtype='bar', rwidth=0.8)
plt.title('price histogram')
plt.figure(2)
plt.title('price')
plt.plot(data)
plt.figure(3)
plt.title('hellinger distance')
plt.plot(distData)
if hits >=10:
  plt.plot([hitx],[threshold],'or')

plt.axhline(distAvg,0,3000,color='r')
plt.axhline(threshold,0,3000,color='g')
plt.show()



