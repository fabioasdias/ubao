from subprocess import call
import time
from glob import glob
import os

MAXIMAGES = 500
IMAGEDIR = './public/'

newImages=[];

while(True):
    time.sleep(10)
    currentImages=sorted(glob(IMAGEDIR+'*jpg'))
    call('instaLooter hashtag art ./public/ -n 50',shell=True)
    newImages=list(set(glob(IMAGEDIR+'*.jpg'))-set(currentImages))
    if len(newImages)==0:
        continue
        
    cSize=len(newImages)+len(currentImages)
    if cSize>MAXIMAGES:
        for i in range(cSize-MAXIMAGES):
            print('\nremoving {0} - {1}'.format(i,currentImages[i]))
            os.remove(currentImages[i]) #lets not run out of space
