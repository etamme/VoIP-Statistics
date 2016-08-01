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
# caHist={'20':[10,29],
#           '75':[50,65],
#           '5' :[90,500]}

cHist={'19':[20,30],
       '30':[30,40],
       '21':[50,70],
       '20':[70,80],
       '8' :[80,90],
       '2' :[100,200]}

# total number of data points
numCalls=5000

# min/max delta for abnormal calls
cDeltaMin=100
cDeltaMax=150

# create abnormal traffic?
abnormal=1

# total number of abnormal calls to create
numACalls=500

#aHist={'70':[150,200],
#           '30':[100,200]}

aHist={'70':[70,90],
       '20':[100,200],
       '10':[200,250]}



# min/max delta for abnormal calls
aDeltaMin=30
aDeltaMax=50

# number of threshold triggers before alert
# hits for hellinger
hHits=10
# hits for mahalanobis
mHits=30
# hits for standard deviation
sHits=10

# calculates euclidean distance between two lists
def hellingerDistance(p, q):
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

# simulate duration of calls based on price
def generateDuration(data,abnormal):
  if abnormal==1:
    rate=1.0
  else:
    rate=.029
  return [v / rate for v in data]

# simulate interarrival delta of calls
def generateDelta(data,minD,maxD):
  return [randint(minD,maxD)/100.0 for v in data]

# simulate a number of call prices given a histogram
def generatePrices(hist,numcalls):
  data=[]
  for k,v in hist.iteritems():
    minVal=int(min(v))
    maxVal=int(max(v))
    percent=float(k)
    calls=numcalls*(percent/100.0)
    #print str(calls)+" between "+str(minVal)+" - "+str(maxVal)
    for c in range(int(calls)):
      val=randint(minVal,maxVal)/100.0
      data.append(val)
  shuffle(data)
  return data

# takes a list and creates a histogram
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
def movingHD(data):
  distData=[]
  window=data[:20]
  dist1=createDistribution(data[:20])
  dist2=list(dist1)
  # calculate the distance on the moving window
  for v in data:
    window.pop(0)
    window.append(v)
    dist2=createDistribution(window)
    distData.append(hellingerDistance(dist1,dist2))
  return distData

# calculate helligers distance on a moving set of 20 calls
def movingMD(data1,data2):
  distData=[]
  window1=data1[:20]
  window2=data2[:20]
  # calculate the distance on the moving window
  for i in range(20,len(data1)):
    window1.pop(0)
    window1.append(data1[i])
    window2.pop(0)
    window2.append(data2[i])
    distData+=mahalanobisDistance(window1,window2)
  return distData

def detectThreshold(data,threshold,hits):
  i=0
  total=0
  for v in data:
    if v >= threshold:
      total+=v
      if total>=hits:
        return i
    if i % 100 == 0:
      total=0
    i+=1
  return 0

def generateStats(data):
  avg=(sum(data)/len(data))
  std=np.std(data)
  trs=avg + (2*std)
  return {'avg':avg,'std':std,'trs':trs}

def spliceLists(data1,data2):
  midIdx=len(data1)/2
  front=data1[0:midIdx]
  end=data1[midIdx+1:]
  midIdx=len(end)/2
  middle=end[0:midIdx]+data2
  shuffle(middle)
  tail=end[midIdx+1:]
  return front+middle+tail


# generate 2 data sets, a list of prices, and a list of interarriaval deltas
# these contain no abnormal traffic.  If we are to generate abnormal traffic,
# we do so later, and splice it into the middle of these data sets
data1=generatePrices(cHist,numCalls)
data2=generateDelta(data1,cDeltaMin,cDeltaMax)

# generate reference distances
hDistRefData1=movingHD(data1)
hDistRefData2=movingHD(data2)
mDistRef=mahalanobisDistance(data1,data2)
# generate reference statistics on data sets
data1RefStats=generateStats(data1)
data2RefStats=generateStats(data2)
hDistRefStatsData1=generateStats(hDistRefData1)
hDistRefStatsData2=generateStats(hDistRefData2)
mDistRefStats=generateStats(mDistRef)

# generate abnormal traffic and splice it into normal traffic
if abnormal==1:
  data3=generatePrices(aHist,numACalls)
  data4=generateDelta(data3,aDeltaMin,aDeltaMax)

  data1=spliceLists(data1,data3)
  data2=spliceLists(data2,data4)

  buckets = [0,1,2,4,9,16,25,36,49]
else:
  buckets = [0.01,0.10,0.20,0.40,0.90,1.6,2.5,3.6,4.9]

# data1 and data2 will have abnormal traffic mixed in at this point if we
# were supposed to generate it, so generate new distances, and statsistics
hDistData1=movingHD(data1)
hDistData2=movingHD(data2)

mDist=mahalanobisDistance(data1,data2)
mDistStats=generateStats(mDist)

data1Stats=generateStats(data1)
data2Stats=generateStats(data2)

hDistStatsData1=generateStats(hDistRefData1)
hDistStatsData2=generateStats(hDistRefData2)

sHitXData1=detectThreshold(data1,data1RefStats['trs'],sHits)
hHitXData1=detectThreshold(hDistData1,hDistRefStatsData1['trs'],hHits)
hHitXData2=detectThreshold(hDistData2,hDistRefStatsData2['trs'],hHits)
mHitX=detectThreshold(mDist,mDistRefStats['trs']+mDistRefStats['std'],mHits)

fig = plt.figure(figsize=(18,10))
fig.suptitle("VoIP Statistics - Eric Tamme")

# data 2
ax = plt.subplot("231")
ax.set_title('hellinger distance on interval')
ax.plot(hDistData2)
if hHitXData2 >0:
  ax.plot([hHitXData2],[hDistRefStatsData2['trs']],'or')
ax.axhline(hDistStatsData2['avg'],0,3000,color='r')
ax.axhline(hDistStatsData2['trs'],0,3000,color='g')


# price
ax = plt.subplot("232")
ax.set_title('price')
ax.plot(data1)

# hellinger distance on price
ax = plt.subplot("233")
ax.set_title('hellinger distance on price')
ax.plot(hDistData1)
if hHitXData1 >0:
  ax.plot([hHitXData1],[hDistRefStatsData1['trs']],'or')
ax.axhline(hDistRefStatsData1['avg'],0,3000,color='r')
ax.axhline(hDistRefStatsData1['trs'],0,3000,color='g')

# mahalanobis distance of data1 and data2
ax = plt.subplot("234")
ax.set_title('mahalanobis distance')
ax.plot(mDist)
ax.axhline(mDistRefStats['avg'],0,3000,color='r')
ax.axhline(mDistRefStats['trs'],0,3000,color='g')
if mHitX >0:
  ax.plot([mHitX],[mDistRefStats['trs']],'or')

# data2
ax = plt.subplot("235")
ax.set_title('interarrival delta')
ax.plot(data2)

# std deviation on data 1
ax = plt.subplot("236")
ax.set_title('standard deviation on price')
ax.plot(data1)
if sHitXData1 >0:
  ax.plot([sHitXData1],[data1RefStats['trs']],'or')
ax.axhline(data1RefStats['avg'],0,3000,color='r')
ax.axhline(data1RefStats['trs'],0,3000,color='g')

# display plot
plt.show()


