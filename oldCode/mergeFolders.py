import sys
from shutil import move
from glob import glob
from os.path import basename

for f in glob(sys.argv[1]+'/*'):
    move(f,sys.argv[2]+'/a_'+basename(f))