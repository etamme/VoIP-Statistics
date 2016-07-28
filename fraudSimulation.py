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
          '8' :[80,90],
          '2' :[100,200]}

# total number of data points
numCalls=5000

# create fraudulent traffic?
fraud=1
# total number of fraudulent calls to create
numFraudCalls=500
#fraudHist={'70':[150,200],
#           '30':[100,200]}

fraudHist={'30':[150,200],
           '70':[100,200]}
# calculates euclidean distance between two lists
def hellinger(p, q):
  return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def mahalanobisDistance(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md


def generateDuration(data,fraud):
  if fraud==1:
    rate=1.0
  else:
    rate=.029
  return [v / rate for v in data]

def generateData(hist,numcalls):
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

def createDistribution(data):
  global numCalls
  data1=[1,2,3,4,5,6,7,8,9,10]
  dist1=[]
  buckets={0:.03,1:.30,2:.40,3:.70,4:.80,5:.90,6:1,7:2,8:4,9:9}
  prev=0
  # create histogram based on buckets for supplied data
  for k,v in buckets.iteritems():
    data1[k]=sum(i > prev and i <=v for i in data[:numCalls/2])
    prev=v
  # normalize the data set
  dist1= [float(i)/sum(data1) for i in data1[:numCalls/2]]
  return dist1


# calculate helligers distance on a moving set of 20 calls
def generateDistance(data):
  distData=[]
  window=data[:20]
  dist1=createDistribution(data[:20])
  dist2=list(dist1)
  # calculate the distance on the moving window
  for v in data:
    window.pop(0)
    window.append(v)
    dist2=createDistribution(window)
    distData.append(hellinger(dist1,dist2))
  return distData

def detectThreshold(data,threshold):
  i=0
  total=0
  for v in data:
    if v >= threshold:
      total+=v
      print "threshold exceeded: "+str(v)+" total: "+str(total)+" index: "+str(i)
      if total>10:
        return i

    if i % 100 == 0:
      total=0
    i+=1
  return 0


# generate data sets
data=generateData(callHist,numCalls)
duration=generateDuration(data,0)
if fraud==1:
  # make a copy of the data before injecting fraud so we have a training set
  nofraud=list(data)
  fraudData=generateData(fraudHist,numFraudCalls)
  fraudDuration=generateDuration(fraudData,1)
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

  # do the same with duration data
  mididx=numCalls/2
  front=duration[0:mididx]
  end=duration[mididx+1:]
  mididx=len(end)/2
  middle=end[0:mididx]+fraudDuration
  shuffle(middle)
  tail=end[mididx+1:]
  duration=front+middle+tail


# create our list of distances
distData=generateDistance(data)
mahaDist=mahalanobisDistance(data,duration)

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

hitx=detectThreshold(distData,threshold)

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
if hitx >=0:
  plt.plot([hitx],[threshold],'or')

plt.axhline(distAvg,0,3000,color='r')
plt.axhline(threshold,0,3000,color='g')
plt.figure(4)
plt.title('mahalanobis distance')
plt.plot(mahaDist)
plt.show()


