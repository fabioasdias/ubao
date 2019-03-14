from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cherrypy
import os
from os.path import basename
#import json
from glob import glob
from random import random
from numpy.linalg import svd,norm

IMAGEDIR='./public/'

def _getXs():
    Xs=[]
    for i in range(20):
        f=glob('./train/{0}.*'.format(i+1))
        print(f)
        Xs.append(_computeFV(f[0]))
    return(np.array(Xs))

def _computeFV(img_path):
    global model
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(model.predict(x).squeeze())

tol = 1.e-6    # zero tolerance

def pdist(x):
    """
    Pairwise distance between pairs of objects
    TODO: find a fast function
    """
    n, d = x.shape
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(x[i] - x[j])
    return dist

#https://github.com/adrianolinux/lamp/blob/master/lamp.py
#thx Adriano! :)
def project(x, xs, ys):
    """
    Projection
    """
    assert (type(x) is np.ndarray) and (type(xs) is np.ndarray) and (type(ys) is np.ndarray), \
        "*** ERROR (Force-Scheme): project input must be numpy.array type."
    x=np.atleast_2d(x)
    ninst, dim = x.shape    # number os instances, data dimension
    k, a = xs.shape         # number os sample instances
    p = ys.shape[1]         # visual space dimension
    
    assert dim == a, "*** LAMP Error: x and xs dimensions must be egual."

    Y = np.zeros((ninst, p))
    for pt in range(ninst):
        # computes alphas
        alpha = np.zeros(k)
        for i in range(k):
            # verify if the point to be projectec is a control point
            # avoids division by zero
            if np.linalg.norm(xs[i] - x[pt]) < tol:
                alpha[i] = np.finfo(float).max
            else:
                alpha[i] = 1 / np.linalg.norm(xs[i] - x[pt])**2
        
        # computes x~ and y~ (eq 3)
        xtilde = np.zeros(dim)
        ytilde = np.zeros(p)
        for i in range(k):
            xtilde += alpha[i] * xs[i]
            ytilde += alpha[i] * ys[i]
        xtilde /= np.sum(alpha)
        ytilde /= np.sum(alpha)

        A = np.zeros((k, dim))
        B = np.zeros((k, p))
        xhat = np.zeros((k, dim))
        yhat = np.zeros((k, p))
        # computation of x^ and y^ (eq 6)
        for i in range(k):
            xhat[i] = xs[i] - xtilde
            yhat[i] = ys[i] - ytilde
            A[i] = np.sqrt(alpha[i]) * xhat[i]
            B[i] = np.sqrt(alpha[i]) * yhat[i]
    
        U, D, V = svd(np.dot(A.T, B)) # (eq 7)
        # VV is the matrix V filled with zeros
        VV = np.zeros((dim, p)) # size of U = dim, by SVD
        for i in range(p): # size of V = p, by SVD
             VV[i,range(p)] = V[i,:]
        M = np.dot(U, VV) # (eq 7)

        Y[pt] = np.dot(x[pt] - xtilde, M) + ytilde # (eq 8)

    return(Y[0][0],Y[0][1])
    

def _getNextImage(imagelist,previous=''):
    previous=basename(previous)
    imagelist=[basename(x) for x in sorted(glob(IMAGEDIR+'*.jpg'))]
    if previous=='':
        return(imagelist[0])
    else:
        try:#faster than if previous in imagelist (methinks)
            curInd=imagelist.index(previous)
            if ((curInd+1)<len(imagelist)):
                return(imagelist[curInd+1])
            else:
                return(imagelist[0])
        except:
            return(imagelist[0])

class server(object):
    imagelist=[]

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def getImage(self,previous=''):
        cherrypy.response.headers["Access-Control-Allow-Origin"] = "*"
        nextImage=_getNextImage(self.imagelist,previous)
        left,top=project(_computeFV(IMAGEDIR+nextImage),Xs,Ys)
        #Ys was multiplied by 100 to be in 0-100 (to avoid numerical errors)
        return({'left':left/100,'top':top/100,'fname':IMAGEDIR+basename(nextImage)})



if __name__ == '__main__':
    global model
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    model._make_predict_function()

    Xs=_getXs()
    Ys=np.array([[225.014,  91.639],
             [201.801,  42.955],
             [136.815, 114.415],
             [127.152, 171.871],
             [257.198,  42.608],
             [202.170,   3.405],
             [260.356,   2.960],
             [253.722, 178.631],
             [194.522, 176.530],
             [252.733, 128.383],
             [195.188, 139.988],
             [ 14.788,  66.296],
             [127.949,  77.293],
             [ 66.252,  68.516],
             [ 21.230, 157.784],
             [  9.625, 112.806],
             [ 76.743, 119.088],
             [ 68.151, 156.527],
             [ 67.536,  18.572],
             [ 29.201,  11.358]])

    Ymin=np.min(Ys,axis=0)
    Ymax=np.max(Ys,axis=0)
    Ys[:,0]=100*((Ys[:,0]-Ymin[0])/(Ymax[0]-Ymin[0]))
    Ys[:,1]=100*((Ys[:,1]-Ymin[1])/(Ymax[1]-Ymin[1]))
    
    print(Ys)
    print(project(Xs[0]+0.2,Xs,Ys))

    conf = {
        '/':{
            'tools.staticdir.root': os.path.abspath(os.getcwd()),
        },
        '/public': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': os.path.abspath(os.getcwd())+'/public'
        }
    }
    
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.server.socket_port = 8080
    cherrypy.tree.mount(server(), '/', conf)

    cherrypy.engine.start()
    cherrypy.engine.block()

