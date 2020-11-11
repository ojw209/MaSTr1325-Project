import os
from matplotlib import pyplot as plt
import random
import numpy as np
import cv2

#Random Seed - Used for splitting data.
random.seed(a = 22031998, version = 2)

#Change Working directory to location of main file. 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% Define class to store data. 
class Img_Class:
    def __init__(self,ID,Flag_T,Flag_V):
        self.ID = ID
        self.Raw_Img = []
        self.Raw_Mask = []
        self.Img_Est_Mask = []
        self.Train_Flag = Flag_T
        self.Validation_Flag = Flag_V


#%% Partition data into training and validation dataset [Note ]
Record_Count = 1
Train_Percent = .8

Training_Indices = sorted(random.sample(range(Record_Count),k = int(Train_Percent*Record_Count) ))
Img_Bank = []

#Split Into training/validation sets.
for i in range(Record_Count):
    
    if i in Training_Indices:
        Img_Bank.append(Img_Class(i,1,0))
        
    else:
        Img_Bank.append(Img_Class(i,0,1))

#%% Read in dataset. [NOTE: Method used for indexing img's maybe unstable at
#   larger data sizes, due to RAM limitations]
Top_Img_Path = 'Images/'
Top_Mask_Path = 'Masks/'


for i in range(Record_Count):
    if (i + 1)  < 10:
        Img_Path = Top_Img_Path + '000' + str(i + 1) + '.jpg'
        Mask_Path  = Top_Mask_Path + '000' + str(i+1) + 'm.png'
        
        Img_Bank[i].Raw_Img = cv2.imread(Img_Path)
        Img_Bank[i].Raw_Mask = cv2.imread(Mask_Path,0)
        
    
    if (i + 1)  >= 10 and (i + 1) < 100 :
        Img_Path = Top_Img_Path + '00' + str(i + 1) + '.jpg'
        Mask_Path  = Top_Mask_Path + '00' + str(i+1) + 'm.png'

        
        Img_Bank[i].Raw_Img = cv2.imread(Img_Path)
        Img_Bank[i].Raw_Mask = cv2.imread(Mask_Path,0)
    
    if (i + 1)  >= 100 and (i + 1) < 1000 :
        Img_Path = Top_Img_Path + '0' + str(i + 1) + '.jpg'
        Mask_Path  = Top_Mask_Path + '0' + str(i+1) + 'm.png'
        

        Img_Bank[i].Raw_Img = cv2.imread(Img_Path)
        Img_Bank[i].Raw_Mask = cv2.imread(Mask_Path,0)
    
    if (i + 1) > 1000 :
        Img_Path = Top_Img_Path + str(i + 1) + '.jpg'
        Mask_Path  = Top_Mask_Path +  str(i+1) + 'm.png'

        Img_Bank[i].Raw_Img = cv2.imread(Img_Path)
        Img_Bank[i].Raw_Mask = cv2.imread(Mask_Path,0)
    


#%% Serialize Photo and place into numpy array ready for learning



j = 0
for i in Training_Indices:
    print(i)
    if j == 0:
        Test_X_R = Img_Bank[i].Raw_Img[:,:,0].ravel()
        Test_X_G = Img_Bank[i].Raw_Img[:,:,1].ravel()
        Test_X_B = Img_Bank[i].Raw_Img[:,:,2].ravel()
        Test_X = np.array((Test_X_R,Test_X_G,Test_X_B)).transpose()
        
        Test_Y = Img_Bank[i].Raw_Mask.ravel()
        
        j = 1
    
    Test_X_R = Img_Bank[i].Raw_Img[:,:,0].ravel()
    Test_X_G = Img_Bank[i].Raw_Img[:,:,1].ravel()
    Test_X_B = Img_Bank[i].Raw_Img[:,:,2].ravel()
    
    Temp_X = np.array((Test_X_R,Test_X_G,Test_X_B)).transpose()
    Temp_Y = Img_Bank[i].Raw_Mask.ravel()
    
    Test_X = np.concatenate((Test_X,Temp_X))
    Test_Y = np.concatenate((Test_Y,Temp_Y))

#%% KNN learning trial.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#classifier = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, verbose )
kernel = 1.0 * RBF(1.0)
classifier = GaussianProcessClassifier(kernel=kernel, random_state=0)

print('KNN - Learning Started')
classifier.fit(Test_X, Test_Y, )
print('KNN - Learning Finished')

print('KNN - Prediction Started')
Test_photo = 10
Test_X_R = Img_Bank[Test_photo].Raw_Img[:,:,0].ravel()
Test_X_G = Img_Bank[Test_photo].Raw_Img[:,:,1].ravel()
Test_X_B = Img_Bank[Test_photo].Raw_Img[:,:,2].ravel()
Test_X = np.array((Test_X_R,Test_X_G,Test_X_B)).transpose()

Y_Pred = classifier.predict(Test_X)
Y_Pred = np.reshape(Y_Pred,(384,512))
plt.matshow(Y_Pred)
plt.gca().xaxis.tick_bottom()
print('KNN - Prediction Finished')



        
# Load an color image in grayscale
img = cv2.imread('Images/0011.jpg')

Truth_img = cv2.imread('Masks/0011m.png',0) 

plt.matshow(Truth_img)
plt.gca().xaxis.tick_bottom()

# Initiate STAR detector
orb = cv2.ORB_create(nfeatures = 500)

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,outImage = None, color=(0,255,0), flags=0)
plt.imshow(img2,plt.show())



