import sys
from PIL import Image
from glob import glob
from libtiff import TIFF
import numpy as np
import matplotlib.pylab as plt
from os import makedirs
from os.path import basename,exists
from shutil import rmtree
from json import dump
class plotStuff:
    def _plotLine(self,x1,y1,x2,y2,style='-r'):
        return(self.ax.plot([x1,x2],[y1,y2],style)[0])

    def _plotRect(self,P1,P2,style='-r'):
        ret=[]
        ret.append(self._plotLine(P1[0],P1[1],P2[0],P1[1]))
        ret.append(self._plotLine(P1[0],P1[1],P1[0],P2[1]))
        ret.append(self._plotLine(P2[0],P2[1],P2[0],P1[1]))
        ret.append(self._plotLine(P2[0],P2[1],P1[0],P2[1]))
        return(ret)

    def __init__(self,panelName,outBaseDir):
        print("""Keybindings:
            x : restart current rectangle
            n : start next rectangle
            w : save results and move to next image
            = : (twice) terminate the program (without saving)
            """)
        self.outBaseDir=outBaseDir
        self.quit=False
        self.panelName=panelName
        tif = TIFF.open(panelName, mode='r')
        panel = tif.read_image()
        if (np.max(panel>255)):
            dmax=65535.0
        else:
            dmax=255.0
        self.panel=np.uint8(255*(panel/dmax)) #16-> 8 bits
        self.crops=[]
        self.curCrop=[]
        self.rects=[]
        self.curRect=[]
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.gray()
        plt.title(basename(panelName))
        self.ax.imshow(self.panel)
        plt.show()


    def onclick(self,event):
        if (not event.dblclick):
            if len(self.curCrop)<=1:
                self.curCrop.append((event.xdata, event.ydata))
                if len(self.curCrop)==2:
                    self.curRect=self._plotRect(self.curCrop[0],self.curCrop[1])
                    self.fig.canvas.draw()

    def press(self,event):
        if event.key == '=':
            if (self.quit):
                exit()
            else:
                self.quit=True
        if event.key == 'x':
            for l in self.curRect:
                l.remove()
            self.curRect=[]
            self.curCrop=[]
            self.fig.canvas.draw()            

        if event.key == 'n':
            for l in self.curRect:
                l.set_color('blue')
            self.rects.append(self.curRect)
            self.crops.append(self.curCrop)
            self.curRect=[]
            self.curCrop=[]
            self.fig.canvas.draw()

        if event.key == 'w':
            outFolder=self.outBaseDir+'/'+basename(self.panelName).replace('.tif','').replace(' ','_')
            if (exists(outFolder)):
                rmtree(outFolder)
            makedirs(outFolder)
            files=[]
            for i,c in enumerate(self.crops):
                fname='im{0}.png'.format(i)
                y1=int(np.floor(c[0][0]))
                x1=int(np.floor(c[0][1]))
                y2=int(np.ceil(c[1][0]))
                x2=int(np.ceil(c[1][1]))
                files.append({'name':fname,'p1':(x1,y1),'p2':(x2,y2)})
                img = Image.fromarray(self.panel[x1:x2,y1:y2])
                img.save(outFolder+'/'+fname)

            print(outFolder+'/pos.json')
            with open(outFolder+'/pos.json','w') as ff:
                dump(files,ff)

            plt.close(self.fig)
    



if len(sys.argv)!=3:
    print('.py inDir outBaseDir')

for f in sorted(glob(sys.argv[1]+'/*.tif')):
    a=plotStuff(f,sys.argv[2])
