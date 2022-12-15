import streamlit as st
import base64
import numpy as np
import cv2
import pywt
import json
import joblib
from base64 import b64decode
from base64 import b64encode
from io import StringIO

__model = None
class celeb_image_classifier():


    def __init__(self,model,bytes_data):
        self.model = model
        self.bytes_data = bytes_data
    

    def w2d(self,img, mode='haar', level=1):
        imArray = img
        #Datatype conversions
        #convert to grayscale
        imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
        #convert to float
        imArray =  np.float32(imArray)
        imArray /= 255;
        # compute coefficients
        coeffs=pywt.wavedec2(imArray, mode, level=level)

        #Process Coefficients
        coeffs_H=list(coeffs)
        coeffs_H[0] *= 0;

        # reconstruction
        imArray_H=pywt.waverec2(coeffs_H, mode);
        imArray_H *= 255;
        imArray_H =  np.uint8(imArray_H)

        return imArray_H


    def convert_image_to_base64(self):
        print(type(self.bytes_data))
        b64str = b64encode(self.bytes_data)
        return b64str


    def get_cv2_image_from_base64_string(self,b64str):
        #b64str = str(b64str)
        #encoded_data = b64str.split(',')[0]
        print(type(b64str))
        nparr = np.frombuffer(b64decode(b64str), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(img.shape)
        return img

    
    def get_cropped_image_if_2_eyes(self,image_path,image_base64_data):
            face_cascade = cv2.CascadeClassifier('./haarCascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('./haarCascades/haarcascade_eye.xml')

            if image_path:
                img = cv2.imread(image_path)
            else:
                img = self.get_cv2_image_from_base64_string(image_base64_data)
                print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print("Gray = ",gray)
            #plt.imshow(gray)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            cropped_faces = []
            for (x,y,w,h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) >= 2:
                        cropped_faces.append(roi_color)
            return cropped_faces
    

    def classify_image(self,image_base64_data, file_path=None):

        imgs = self.get_cropped_image_if_2_eyes(file_path, image_base64_data)
        if(imgs == []):
            
            return None
        result = []
        for img in imgs:
            scalled_raw_img = cv2.resize(img, (32, 32))
            print("Array Shape",scalled_raw_img.shape)
            img_har = self.w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (64, 64))
            print("Array Shape",scalled_img_har.shape)
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32 * 4, 1)))

            len_image_array = 32*32*3 + 32*32*4

            final = combined_img.reshape(1,len_image_array).astype(float)

        return self.model.predict(final)
    

    def load_saved_artifacts():
        print("loading saved artifacts...start")
        global __class_name_to_number
        global __class_number_to_name

        #with open("/Users/pratikraj/Desktop/ImageClassification/artifacts/celeb_name_number.json", "r") as f:
         #   __class_name_to_number = json.load(f)
          #  __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
        print("loading saved artifacts...done")
        return __model


    def convert_number_to_name(self,predicted_number):

        with open('./artifacts/celeb_name_number.json') as json_file:
            data = json.load(json_file)
        name = data[str(predicted_number)]["name"]
        url = data[str(predicted_number)]["url"]
        return [name,url]
        #return name
        

if __name__ == "__main__":

    st.title('Image Classification')
    uploaded_file = st.sidebar.file_uploader("Please Choose the Picture")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        model = celeb_image_classifier.load_saved_artifacts()
        obj = celeb_image_classifier(model,bytes_data)
        base64_data = obj.convert_image_to_base64()
        #print(type(base64))
        answer = obj.classify_image(base64_data,None)
        if(answer is not None):
            print("Answer is ",answer)
            list = obj.convert_number_to_name(answer[0])
            name = list[0]
            url = list[1]
            st.write("This Picture is of ",name)
            st.write(f"[Click](%s) to get more details of {name}" % url)
            img = obj.get_cv2_image_from_base64_string(base64_data)
            st.image(img)
            #st.write("[Click] to Get more Details of the Person in the picture(%s)" % url)
        else:
            st.write("Sorry the Image was not clear enough to recognize")

    

    
