import numpy as np
import os

#np.set_printoptions(threshold=np.inf)
import cv2
import dlib

# For dlib face thing
from imutils import face_utils
import argparse
import imutils

#For uploading Files
from os import listdir
from os.path import isfile, join

#For creating a CSV
import csv

#To extract gender
import re

#for KNN
import pandas as pd  
import operator
import math
 

# =============== Function for uploading all files (3253 Files) ==========================
def fileParser():
	onlyfiles = [f for f in listdir('./Test_Data') if isfile(join('./Test_Data', f))]
	
	for i in onlyfiles:
		faceDetection('./Test_Data/' + i)
# =============== END Uploading Files  ==========================


# =============== Function for CSV ==========================
def createCSV(feature_vector):
	with open('test.csv', 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([ feature_vector[0], feature_vector[1], feature_vector[2], feature_vector[3], feature_vector[4], feature_vector[5], feature_vector[6], feature_vector[7]])

# =============== END CSV  ==========================


# =============== Function for KNN Below ==========================
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
 
    length = testInstance.shape[1]
    
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]

    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
 
    neighbors = []
    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}
    
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
 
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)
# ========================== End of KNN =============================

# =============== Function for Image cropping Below ==========================
def facecrop(image):
	print("Cropping Image... \n")
	facedata = "haarcascade_frontalface_alt.xml"
	cascade = cv2.CascadeClassifier(facedata)

	img = cv2.imread(image)

	minisize = (img.shape[1],img.shape[0])
	miniframe = cv2.resize(img, minisize)

	faces = cascade.detectMultiScale(miniframe)

	for f in faces:
		x, y, w, h = [ v for v in f ]
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

		sub_face = img[y:y+h, x:x+w]
		fname, ext = os.path.splitext(image)
		cv2.imwrite(fname+"_cropped_"+ext, sub_face)


	original_file_name = image.split('.')[0]
	return(original_file_name + "_cropped_.jpg")
# ========================== End image cropping =============================

#Function that uploads picture to the algorithm
def picUploader():
	fileName = facecrop("Amari.jpg")
	return fileName


#Function that does the facial detection 
def faceDetection(img):
	PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	predictor = dlib.shape_predictor(PREDICTOR_PATH)
	detector = dlib.get_frontal_face_detector()

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	k = 0

	features = []
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

	# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# To display output to the console
			arr = ["Mouth", "Right Eyebrow", "Left Eyebrow", "Right Eye", "Left Eye", "Nose", "Jaw", "Complete Face w/ Mapping"]
			#print("Conducting Facial Feature Detection For " + arr[k] + "...\n")
			k +=1

			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

			#Variable to create the feature vector
			feat_vec = roi
			#Will use standard deviation of the ROI for each region
			features.append(np.mean(feat_vec))

			# show the particular face part
			#cv2.imshow("ROI", roi)
			#cv2.imshow("Image", clone)
			#cv2.waitKey(0)

		# visualize all facial landmarks with a transparent overlay
		output = face_utils.visualize_facial_landmarks(image, shape)
		#cv2.imshow("Image", output)
		#cv2.waitKey(0)

		#Adds gender for the CSV File
		'''
	
		gender = re.findall(r"[\w']+", img)
		gender = gender[1]
		gender = gender.split('_')[1]

		features.append(int(gender))
		'''
		
		return(features)

		#To create the csv from the train data
		#createCSV(features)
	
#Function to train the data
def trainData():
	fileParser()

#Function to run algorithm on unseen image
def testData():
	image = picUploader()

	data = pd.read_csv("test2.csv")

	#Creates test data for unseen image
	testSet = [faceDetection(image)]
	test = pd.DataFrame(testSet)

	print(testSet)

	# Setting number of neighbors = 1
	k = 5
	
	# Running KNN model
	result,neigh = knn(data, test, k)

	
	if(result == 0):
		print("This is a male")
	else:
		print("This is a female")

	# Predicted class
	#print(result)

	# Nearest neighbor
	#print(neigh)



#The main function of the program
def main():
	#====To Train===
	#trainData()
	#===END Train===

	#===To Test===
	testData()
	#===END Test===
	#createCSV()
	

main()