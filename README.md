## Deep-learning Radiomics techniques for classification modelling

### Introduction
This is the code for the paper entitled "Computed tomography-based deep-learning prediction of neoadjuvant chemoradiotherapy treatment response in esophageal squamous cell carcinoma"

This code includes feature extraction from pretrained Deep Convolutional Neural Network model and then further model training and validation by machine learning approaches.

### Methods
Analysis flowchart.
- Radiological features extracted from the deep learning method and handcrafted radiomics method
- Machine learning methods in model construction
- Model evaluation

### Requirement
All requirements are given in ```requirements.txt```
#### Python requirements
- numpy 1.16.3
- pandas 0.24.2
- scipy 1.2.1
- SimpleITK 1.2.0
- Keras 2.2.4
- scikit-learn 0.21.1


#### R requirements
- psych 1.8.12
- combat 2.0
- data.table 1.12.6

### Performance
```
The followed table showed the area under the receiver operating characteristic curve (AUC) by
different feature extractorsin the external test cohort
```
| Method | AUC |
|:---|:---:|
| ResNet50 | 0.805 |
| Xception | 0.763 |
| VGG16 | 0.648 |
| VGG19 | 0.635 |
| InceptionV3 | 0.753 |
| InceptionResNetV2 | 0.653 |
| Radiomics | 0.725 |

```
The feature maps generated from ResNet50 indicated locations that were important for output
generation (followed figure). Tumoral and peri-tumoral areas of the images were shown to be 
valuable for the feature pattern extraction.
```
![Suppl fig 3](https://user-images.githubusercontent.com/63107895/78547275-6d421780-7831-11ea-9002-d9319cafc369.jpg)


### Reference

[
Xception: Deep Learning with Depthwise Separable Convolutions
](https://arxiv.org/abs/1610.02357)

[
Very Deep Convolutional Networks for Large-Scale Image Recognition
](https://arxiv.org/abs/1409.1556)

[
Deep Residual Learning for Image Recognition
](https://arxiv.org/abs/1512.03385)

[
Rethinking the Inception Architecture for Computer Vision
](https://arxiv.org/abs/1512.00567)

[
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
](https://arxiv.org/abs/1602.07261)

[
Visual Explanations from Deep Networks via Gradient-Based Localization
](https://arxiv.org/abs/1610.02391)

[Harmonization of multi-site imaging data with ComBat
](https://github.com/Jfortin1/ComBatHarmonization)


