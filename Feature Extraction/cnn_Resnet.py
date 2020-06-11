import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from keras.preprocessing import image
 
## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet', include_top=False)

# Load batch file
imgDir = '.../ESCC_ML-master/example/images'
dirlist = os.listdir(imgDir)[1:]
print(dirlist)

# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg
# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)    
    return img

# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)
        
def featureextraction(imageFilepath,maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath) 
    mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)    
    
    x = image.img_to_array(roi_images2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model_pool_features = model.predict(x)
    
    feature_map = base_model_pool_features[0]
    feature_map = feature_map.transpose((2,1,0))
    features = np.max(feature_map,-1)
    features = np.max(features,-1)
    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures
 
featureDict = {}
for ind in range(len(dirlist)):
    path = os.path.join(imgDir,dirlist[ind])

    seg = loadSegArraywithID(path,'seg')
    im = loadImgArraywithID(path,'im')
        
    deeplearningfeatures = featureextraction(im,seg) 

    result = deeplearningfeatures
    key = list(result.keys())
    key = key[0:]
        
    feature = []
    for jind in range(len(key)):
        feature.append(result[key[jind]])
        
    featureDict[dirlist[ind]] = feature
    dictkey = key
    print(dirlist[ind])
    
dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
dataframe.to_csv('.../ESCC_ML-master/Expected results/Features_Resnet.csv')