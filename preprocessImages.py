import sys
from subprocess import call
from os.path import exists
from tqdm import tqdm
from multiprocessing import Pool

def _doThings(inName):
   outName='out_{0}'.format(inName)
   if not exists(outName):
      call('convert "{0}" -resize 400 {1}'.format(inName,outName),shell=True)


with open(sys.argv[1],'r') as fin:
   todo=fin.read().split('\n')
   p=Pool()
   p.map(_doThings,todo)
   p.close()

         
