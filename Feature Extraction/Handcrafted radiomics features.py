import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd

from radiomics import featureextractor
#van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. `https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>`_

## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load batch file
imgDir = '.../ESCC_ML-master/example/images'
dirlist = os.listdir(imgDir)[1:]
print(dirlist)


# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('seg3' in i.lower()) & (iden in i.lower())][0]
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
featureDict = {}
for ind in range(len(dirlist)):
    path = os.path.join(imgDir,dirlist[ind])
    
    # you can make your own pipeline to import data, but it must be SimpleITK images
    mask = loadSegArraywithID(path,'seg3')
    img = loadImgArraywithID(path,'i')
    params = '.../ESCC_ML-master/Feature Extraction/Paramsescc.yaml'
    
    extractor = featureextractor.RadiomicsFeaturesExtractor(params)

    result = extractor.execute(img,mask)
    key = list(result.keys())
    key = key[1:] 
    
    feature = []
    for jind in range(len(key)):
        feature.append(result[key[jind]])
        
    featureDict[dirlist[ind]] = feature
    dictkey = key
    print(dirlist[ind])
    
dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
dataframe.to_csv('.../ESCC_ML-master/Expected results/Features_Radiomics.csv')