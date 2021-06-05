import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# for video face detection

cap = cv2.VideoCapture(0)

while True:
	_, img = cap.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	
	for x, y, w, h  in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
	
	cv2.imshow('img', img)
	
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break
		

cap.release()


'''
# for image face detection

img = cv2.imread('download.jfif')

resized = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for x, y, w, h in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
cv2.imshow('img', resized)
cv2.waitKey(0)

'''
