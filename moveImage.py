from shutil import move
from glob import glob
import sys
from os.path import basename

for i,f in enumerate(glob(sys.argv[1]+'/**/*.jpeg')):
    move(f,sys.argv[1]+'/real_{0}.jpeg'.format(i))
