from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from glob import glob
import sys
import numpy as np

def _computeFV(img_path):
    global model
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(model.predict(x).squeeze())


if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('.py folder')
        exit(-1)
    
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    model._make_predict_function()

    for im in sorted(glob(sys.argv[1]+'/**/t_*.png')):
        print(im)
        np.savetxt(im.replace('.png','.fv'),_computeFV(im))
