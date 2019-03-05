from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from glob import iglob
import sys
import numpy as np
from os.path import exists, join

def _computeFV(img_path):
    global model
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    np.savetxt( img_path.replace(ext,'.fv'), model.predict(x).squeeze() )


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('.py folder files.txt')
        exit(-1)
    
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    model._make_predict_function()
    
    #    ext='.png'
    #    to_do=glob(sys.argv[1]+'/**/t_*.png')
    #    if (not to_do):
    #        to_do=glob(sys.argv[1]+'/**/*.png')
    #    if (not to_do):

    ext='.jpg'
    with open(sys.argv[2],'r') as files_in:
        for line in files_in:
            im=join(sys.argv[1],line.strip())
            if not exists(im.replace(ext,'.fv')):
                _computeFV(im)
