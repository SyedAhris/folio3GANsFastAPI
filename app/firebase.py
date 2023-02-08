import pyrebase
import os
from datetime import datetime
firebaseConfig = {
}

def initFirebase():
    return pyrebase.initialize_app(firebaseConfig)

def getStorage(firebase):
    return firebase.storage()

def uploadImages(firebase, ip):
  storage = getStorage(firebase)
  images = os.listdir('./images')
  currDateTime = datetime.now()
  strDate = currDateTime.strftime("%d:%b:%Y")
  strTime = currDateTime.strftime("%H:%M:%S.%f")
  cloudDirectory = '/images/'+ip+'-'+strDate+'-'+strTime+'/'
  print(images)
  for image in images:
    storage.child(cloudDirectory+image).put('./images/'+image)
  return cloudDirectory
