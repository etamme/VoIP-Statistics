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
# the data values are the keys, the percent is the value
# NOTE data values WIlL BE DIVIDED BY 100
# this is so we can use randInt and still end up with data values like 0.02
#
# EXAMPLE
# 0-.02 = 25%,  .02-.30 = 75%, .30-5.00 = 5%
# callHist={'2':25,'30':70,'500':5}
callHist={'2':30,'30':70}

# total number of data points
numCalls=5000

# create fraudulent traffic?
fraud=1
# total number of fraudulent calls to create
fraudCalls=500
# minimum and maximum value of fraud data points
# NOTE this will be divided by 100
fraudMin=75
fraudMax=1500


# calculates euclidean distance between two lists
def hellinger(p, q):
  return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


# generate data set
keys=callHist.keys()
keys=map(int,keys)
keys.sort()
data=[]
for i in range(len(keys)):
  if i == 0:
    minVal=1
  else:
    minVal=int(keys[i-1])

  maxVal=int(keys[i])

  percent=float(callHist[str(keys[i])])
  calls=numCalls*(percent/100.0)
  for c in range(int(calls)):
    val=randint(minVal,maxVal)/100.0
    data.append(val)


# randomize the calls since they have been generated in order of histogram keys
shuffle(data)

if fraud==1:
  # make a copy of the data before injecting fraud so we have a training set
  nofraud=list(data)
  fraudData=[]
  for c in range(fraudCalls):
      val=randint(fraudMin,fraudMax)/100.0
      fraudData.append(val)

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

# calculate helligers distance on a moving set of 20 calls
dist1= [float(i)/sum(data[:20]) for i in data[:20]]
dist2=list(data[:20])
distData=[]
for v in data:
  dist2.pop(0)
  dist2.append(v)
  distData.append(hellinger(dist1,[float(i)/sum(dist2) for i in dist2]))

if fraud==1:
  buckets = [0,1,2,4,9,16,25,36,49]
  dist2=list(nofraud[:20])
  fraudDistData=[]
  for v in nofraud:
    dist2.pop(0)
    dist2.append(v)
    fraudDistData.append(hellinger(dist1,[float(i)/sum(dist2) for i in dist2]))
  distAvg=sum(fraudDistData)/len(fraudDistData)
  distStd=np.std(fraudDistData)

else:
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



