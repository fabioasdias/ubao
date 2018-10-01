from glob import glob
import sys
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist

if len(sys.argv)!=2:
    print('.py folder')

def _flips(im):
    ret=[]
    ret.append(im)
    ret.append(np.flipud(im))
    ret.append(np.fliplr(im))
    ret.append(np.flipud(np.fliplr(im)))
    return(ret)

def _warps(im):
    ret=[]
    ret.append((255*warp(im, s1.inverse)).astype(np.uint8))
    ret.append((255*warp(im, s2.inverse)).astype(np.uint8))
    return(ret)


s1 = AffineTransform(shear=-0.12)
s2 = AffineTransform(shear=0.12)


for f in glob(sys.argv[1]+'/**/im*.png'):
    print(f)
    im = (255*equalize_adapthist(np.array(Image.open(f)))).astype(np.uint8)

    base=[]
    base.append(im[0:int(im.shape[0]*0.75),
                   0:int(im.shape[1]*0.75)])
    base.append(im[int(im.shape[0]*0.25):,
                   int(im.shape[1]*0.25):])
    base.append(im[0:int(im.shape[0]*0.75),
                   int(im.shape[1]*0.25):])
    base.append(im[int(im.shape[0]*0.25):,
                   0:int(im.shape[1]*0.75)])

    to_use=[]
    for b in base:
        to_use.extend(_flips(b))
        to_use.extend(_warps(b))
    
    for i,img in enumerate(to_use):
        if (len(img.shape)==2):
            rgbImage=np.stack((img,img,img),axis=2).astype(np.uint8)
        else:
            rgbImage=img

        pilImg=Image.fromarray(rgbImage)
        pilImg.save(f.replace('im','t_im').replace('.png','_{0}.png'.format(i)))