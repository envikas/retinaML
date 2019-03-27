from tkinter import Tk, ttk, Frame, Button, Label, Entry, Text, Checkbutton, \
    Scale, Listbox, Menu, BOTH, RIGHT, RAISED, N, E, S, W, \
    HORIZONTAL, END, FALSE, IntVar, StringVar, messagebox as box
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
result = None


class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master, background="#141e2b")
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        global result

        # changing the title of master widget
        self.master.title("Retinopathy")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # Title - Test for Diabetic Retinopathy
        titleLabel = Label(self, text="Test for Diabetic Retinopathy", font=('', 15), bg="#141e2b", fg="white")
        titleLabel.place(x=275, y=15)

        leftEyeLabel = Label(self, text="Left Eye Image", bg="#141e2b", fg="white")
        leftEyeLabel.place(x=125, y=70)

        leftImageBtn = Button(self, text='Browse', command=self.leftImageBtnClick)
        leftImageBtn.place(x=140, y=100)

        rightEyeLabel = Label(self, text="Right Eye Image", bg="#141e2b", fg="white")
        rightEyeLabel.place(x=575, y=70)

        leftImageBtn = Button(self, text='Browse', command=self.rightImageBtnClick)
        leftImageBtn.place(x=600, y=100)

        # creating a button instance
        analyzeButton = Button(self, text="Analyze", command=self.client_analyze)

        # placing the button on my window
        analyzeButton.place(x=375, y=450)




    def client_analyze(self):
        global imLOriginal
        global imROriginal
        global result
        # call the method to test the image
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
        model = load_model('../models/DR_Five_Classes_recall_0.7796.h5')
        X_test = imageArray
        X_test = self.reshape_data(X_test, img_rows, img_cols, channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        y_pred = model.predict_classes(X_test)
        print(model.predict(X_test))
        lefteyeresult = ''
        righteyeresult = ''
        if y_pred[0] == 0:
            lefteyeresult = 'No Retinopathy'
        elif y_pred[0] > 0:
            lefteyeresult = 'Possible Retinopathy'
        if y_pred[1] > 0:
            righteyeresult = 'Possible Retinopathy'
        elif y_pred[1] == 0:
            righteyeresult = 'No Retinopathy'
        returnText = 'Left Eye Result: ' + lefteyeresult + "\nRight Eye Result: " + righteyeresult
        result = Label(self, text=returnText, anchor='w', bg="#141e2b", fg="white")
        result.place(x=300, y=500)

    def leftImageBtnClick(self):
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
        leftImage=Label(self, width="250", height="250", image = tkimage)
        leftImage.image = tkimage
        leftImage.place(x=50, y=140)

    def rightImageBtnClick(self):
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

        rightImage=Label(self, width="250", height="250",image = tkimage)
        rightImage.image = tkimage
        rightImage.place(x=500, y=140)

    def reshape_data(self, arr, img_rows, img_cols, channels):
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

def main():
    root = Tk()

    # size of the window
    root.geometry("800x600")

    app = Window(root)

    root.mainloop()

if __name__ == '__main__':
    main()
