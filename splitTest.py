from glob import glob
from shutil import move
import sys
from os.path import join,basename,split,exists
from os import makedirs

if len(sys.argv)!=3:
    print('.py origin dest')
    exit(-1)

O=sys.argv[1]
D=sys.argv[2]

for f in glob(join(O,'**/t_im4*')): #im1 and im4
    panel=basename(split(f)[0])
    cd=join(D,panel)
    if (not exists(cd)):
        makedirs(cd)
    # print(f,join(cd,basename(f)))
    move(f,join(cd,basename(f)))

