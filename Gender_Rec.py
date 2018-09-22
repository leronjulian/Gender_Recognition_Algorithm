import numpy as np
import cv2


# =============== Stuff for Image cropping Below ==========================
import os

def facecrop(image):
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

#facecrop("Obama.jpg")

# ========================== End image cropping =============================

#Function that uploads picture to the algorithm
def picUploader():
	fileName = facecrop("Obama.jpg")
	image = cv2.imread(fileName)

	return image


#Function to write the pic to a file
def writePic(img):
	#Writes the image with the features detected
	cv2.imwrite( "./face_detect.jpg", img)
	#cv2.waitKey(0) #Press 0 to close screen
	#cv2.destroyAllWindows()
	print('Detected faces')

#Function that does the facial detection and puts boxes around the features
def faceDetection(img):
	
	#XML files used to get the face and eye detection (Must be in the same folder as the project)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



	#Converts the image to gray because it handles things better.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)

	    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	
	writePic(img)


#Crop the image
def main():
	image = picUploader()

	faceDetection(image)

main()