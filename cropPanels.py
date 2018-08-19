import sys
from PIL import Image
from libtiff import TIFF
import numpy as np
from scipy.ndimage import prewitt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pylab as plt

if len(sys.argv)!=2:
    print('.py .tif')
panelName=sys.argv[1]

tif = TIFF.open(panelName, mode='r')
panel = tif.read_image()
panel=gaussian_filter(np.uint8(255*(panel/np.max(panel))),10)

plt.gray()
plt.imshow(prewitt(panel))
plt.show()



# img = Image.fromarray(np.uint8(255*(panel/np.max(panel))))
# img.save(panelName.replace('.tif','.png'))
