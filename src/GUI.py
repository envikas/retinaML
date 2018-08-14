from tkinter import *
from PIL import Image, ImageTk
import tkinter.filedialog

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np

from keras.models import load_model


imLOriginal = None
imROriginal = None
def leftImageBtnClick():
    path=tkinter.filedialog.askopenfilename()
    im1 = Image.open(path)
    global imLOriginal
    imLOriginal = io.imread(path)
    # Resizing to 250 for window display
    im1 = im1.resize((250, 250), Image.ANTIALIAS)

    # Resizing for NN
    cropx = 1800
    cropy = 1800
    y, x, channel = imLOriginal.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    img = imLOriginal[starty:starty + cropy, startx:startx + cropx]
    imLOriginal = resize(img, (256, 256))
    tkimage = ImageTk.PhotoImage(im1)
    leftImage=Label(window, width="250", height="250", image = tkimage)
    leftImage.image = tkimage
    leftImage.grid(row=3, column=0, pady=10)


def rightImageBtnClick():
    path=tkinter.filedialog.askopenfilename()
    im2 = Image.open(path)
    global imROriginal
    imROriginal = io.imread(path)
    im2 = im2.resize((250, 250), Image.ANTIALIAS)

    # Resizing for NN
    cropx = 1800
    cropy = 1800
    y, x, channel = imROriginal.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    img = imROriginal[starty:starty + cropy, startx:startx + cropx]
    imROriginal = resize(img, (256, 256))
    tkimage = ImageTk.PhotoImage(im2)
    
    rightImage=Label(window, width="250", height="250",image = tkimage)
    rightImage.image = tkimage
    rightImage.grid(row=3, column=1,pady=10)

def analyze():
    global imLOriginal
    global imROriginal
    #call the method to test the image
    # Converting images to arrays
    io.imsave('../data/test-resized-256_single/imageleft.jpeg', imLOriginal)
    imLOriginal = Image.open('../data/test-resized-256_single/imageleft.jpeg')
    io.imsave('../data/test-resized-256_single/imageright.jpeg', imROriginal)
    imROriginal = Image.open('../data/test-resized-256_single/imageright.jpeg')
    imageArray = []
    imageArray.append(np.array(imLOriginal))
    imageArray.append(np.array(imROriginal))
    imageArray = np.array(imageArray)
    # Passing the image through the NN model
    img_rows, img_cols = 256, 256
    channels = 3
    model = load_model('../models/DR_Five_Classes_recall_1.0.h5')
    X_test=  imageArray
    X_test = reshape_data(X_test, img_rows, img_cols, channels)
    X_test = X_test.astype('float32')
    X_test /= 255
    y_pred = model.predict_classes(X_test)
    print(y_pred)
    lefteyeresult = ''
    righteyeresult = ''
    if y_pred[0] == 0:
       lefteyeresult = 'Cat'
    elif y_pred[0]== 1:
        lefteyeresult = 'Dog'
    if y_pred[1] == 1:
        righteyeresult = 'Dog'
    elif y_pred[1] == 0:
        righteyeresult = 'Cat'
    returnText = 'Left Eye Result: ' + lefteyeresult  + "       Right Eye Result: " + righteyeresult
    result.config(text=returnText)

def reshape_data(arr, img_rows, img_cols, channels):
    '''
    Reshapes the data into format for CNN.

    INPUT
        arr: Array of NumPy arrays.
        img_rows: Image height
        img_cols: Image width
        channels: Specify if the image is grayscale (1) or RGB (3)

    OUTPUT
        Reshaped array of NumPy arrays.
    '''
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)
    
window = Tk()

window.title('Retinopathy')
window.geometry("800x600") 
window.resizable(1, 1) #Don't allow resizing in the x or y direction

leftEyeLabel = Label(window, text="Left Eye Image")
leftEyeLabel.grid(row=0, column=0, ipadx=50)

leftImageBtn = Button(window, text='Browse', command=leftImageBtnClick)
leftImageBtn.grid(row=1, column=0)

rightEyeLabel = Label(window, text="Right Eye Image")
rightEyeLabel.grid(row=0, column=1, ipadx=100, pady=10)

rightImageBtn = Button(window, text='Browse', command=rightImageBtnClick)
rightImageBtn.grid(row=1, column=1)

leftImageBtn11 = Button(window, text='Analyze', command=analyze)
leftImageBtn11.grid(row=5, column=0, padx=(400,1), pady=(50,1))

result = Label(window, text="asdadasds")
result.grid(row=6, column=0, padx=(400,1), pady=20)

window.mainloop()
