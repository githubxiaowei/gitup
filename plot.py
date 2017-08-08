import numpy as np
import matplotlib.pyplot as plt
import os


epi, score, Q  = [], [], []

os.chdir('/home/xiaowei/Download/gitdown/gitup')
os.system('git pull --update')

with open('/home/xiaowei/Download/gitdown/gitup/log','r') as f:
	i = 0
	for s in f:
		i += 1
		if i > 34:
			list = s.split(' ')
			epi.append(list[1])
			score.append(list[4])
			Q.append(list[11].split(':')[1])

print(epi[-1])
f1 = plt.figure(1)
f1.add_subplot(1,2,1)
plt.plot(epi,score,'.')
f1.add_subplot(1,2,2)
plt.plot(epi,Q,'.')

epi, score, Q  = [], [], []
with open('/home/xiaowei/Download/gitdown/gitup/log12','r') as f:
	i = 0
	for s in f:
		i += 1
		if i > 34:
			list = s.split(' ')
			#print(list)
			epi.append(list[1].split(':')[1])
			score.append(list[2].split(':')[1])
			Q.append(list[7].split(':')[1].replace('\n',''))

print(epi[-1])
f2 = plt.figure(2)
f2.add_subplot(1,2,1)
plt.plot(epi,score,'.')
f2.add_subplot(1,2,2)
plt.plot(epi,Q,'.')
plt.show()



