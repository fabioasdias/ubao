from glob import glob
from subprocess import call

fileList=glob('*.jpg')

for i,f in enumerate(fileList):
    call('convert "{0}" -resize 400 ./ui/public/img/i{1}.jpg'.format(f,i),shell=True)
