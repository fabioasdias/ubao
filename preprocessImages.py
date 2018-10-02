from glob import glob
from subprocess import call

fileList=glob('*.jpeg')

for i,f in enumerate(fileList):
    call('convert "{0}" -resize 400 out_{1}.jpg'.format(f,i),shell=True)
