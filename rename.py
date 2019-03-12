import sys
from os import rename
from tqdm import tqdm

with open(sys.argv[1]) as fin:
    files=fin.read().split('\n')
for f in tqdm(files):
    fs=f.strip()
    rename(fs,fs[4:])