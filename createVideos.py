import skvideo.io
import numpy as np
from tqdm import tqdm
from numpy.random import choice,randint
import numpy as np
from PIL import Image
from os.path import join, exists
import sys


if len(sys.argv)!=2:
    todoName='todo.txt'
else:
    todoName=sys.argv[1]


W=1920
sW=200
NumW=int(np.floor(W/sW))
gapW=int(np.floor( (W-NumW*sW)/(NumW-1)))

H=1080
sH=150
NumH=int(np.floor(H/sH))
gapH=int(np.floor( (H-NumH*sH)/(NumH-1)))

total=NumH*NumW


def _h2p(hist):
    """From a histogram of previous occurences to p weights to be 
    used in choice that give more probability to less picked values"""
    p=(np.max(hist)*np.ones_like(hist))-hist
    if (np.max(p)==0):
        p=np.ones_like(p)/len(p)
    else:
        p=p/np.sum(p)
    return(p)

grid=[]
for i in range(NumH):
    for j in range(NumW):
        h=int(i*(sH+gapH))
        w=int(j*(sW+gapW))
        grid.append([h,w])


#10 minutes at 25frames/second 
numFrames=10*60*25 #10*60

with open(todoName,'r') as fin:
    panels=[x.strip() for x in fin.read().split('\n') if x.strip()!='']

# panels=glob('Panel*.txt')

for panel in panels:
    print(panel)
    vidName=panel.replace('.txt','.mp4')
    if exists(vidName):
        continue

    with open(panel) as fin:
        imagelist=[x.strip() for x in fin.read().split('\n') if x.strip()!='']

    N=len(imagelist)

    n_to_add=min([N/(numFrames-1),total/2])
    print('adding {0} per frame'.format(n_to_add))

    uses=np.zeros((N,),dtype=np.int)

    to_use=-np.ones((numFrames,total),dtype=np.int)
    cummulative=0
    print('picking images')
    for i in tqdm(range(numFrames)):
        cummulative+=n_to_add
        if (cummulative>1):
            cN=int(np.floor(cummulative))
            cummulative-=cN

            newImages=[]
            while len(newImages)<cN:
                minUsed=np.min(uses)
                possibles=list(np.atleast_1d(np.squeeze(np.where(uses==minUsed))))

                if (len(newImages)+len(possibles))>cN:
                    possibles=possibles[:(cN-len(newImages))]
                for img in possibles:
                    uses[img]+=1
                
                newImages.extend(possibles)
                
            for img in newImages:
                ind=choice(list(range(total)),size=(1,),p=_h2p(np.sum(to_use!=-1,axis=0)))
                to_use[i,ind]=img

    print('duplicates')

    for j in tqdm(range(total)):
        if (np.max(to_use[:,j]))==-1:
            to_use[:,j]=choice(list(range(N)),size=(1,),p=_h2p(uses))
            uses[to_use[0,j]]+=1
        else:
            used=list(np.atleast_1d(np.squeeze(np.where(to_use[:,j]!=-1))))
            to_use[used[-1]:,j]=to_use[used[-1],j]
            to_use[:used[0]:,j]=to_use[used[-1],j]
            for i in range(1,len(used)):
                to_use[used[i-1]:used[i],j]=to_use[used[i],j]

    # for i in range(numFrames):
    #     print(i,np.any(to_use[i,:]>N),to_use[i,:],N)
    #     print(i,np.squeeze(to_use[i]).astype(np.int))
    #     input('.')
            
    print('least used image was used {0}, most {1}'.format(np.min(uses),np.max(uses)))


    print('assembling video')
    writer = skvideo.io.FFmpegWriter(vidName, 
        outputdict={'-vcodec': 'libx264'})

    frame = np.zeros((H, W, 3),dtype=np.uint8)
    for i in tqdm(range(numFrames)):
        for j,img in enumerate(to_use[i].squeeze()):
            if (i==0) or (to_use[i,j]!=to_use[i-1,j]):
                # try:
                newIm=np.array(Image.open(join('./img/',imagelist[img])).resize((sW,sH),Image.BILINEAR))
                # except:
                #     print(img,len(imagelist))
                #     exit()
                if (len(newIm.shape)==2):
                    newIm=np.stack([newIm,newIm,newIm],axis=2)            
                frame[grid[j][0]:grid[j][0]+sH,grid[j][1]:grid[j][1]+sW,:]=newIm[:,:,0:3]#RGBA
        writer.writeFrame(frame)            
    writer.close()

    # vid = vid.astype(np.uint8)


