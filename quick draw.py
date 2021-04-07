from tkinter import *
import pyttsx3
import PIL.ImageOps
from PIL import Image
import numpy as np
from PIL import EpsImagePlugin
import tensorflow as tf
#import matplotlib.pyplot as plt
import threading
oldtext=""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices)>0:
 config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
EpsImagePlugin.gs_windows_binary =  r'bin\gswin64c'


modelfilename="model1"
labellist=["The Eiffel Tower","The Great Wall of China","The Mona Lisa","aircraft carrier","airplane","alarm clock","ambulance","angel","animal migration","ant","anvil","apple","arm","asparagus","axe","backpack","banana","bandage","barn","baseball bat","baseball","basket","basketball","bat","bathtub","beach","bear","beard","bed","bee","belt","bench","bicycle","binoculars","bird","birthday cake","blackberry","blueberry","book","boomerang","bottlecap","bowtie","bracelet","brain","bread","bridge","broccoli","broom","bucket","bulldozer","bus","bush","butterfly","cactus","cake","calculator","calendar","camel","camera","camouflage","campfire","candle","cannon","canoe","car","carrot","castle","cat","ceiling fan","cell phone","cello","chair","chandelier","church","circle","clarinet","clock","cloud","coffee cup","compass","computer","cookie","cooler","couch","cow","crab","crayon","crocodile","crown","cruise ship","cup","diamond","dishwasher","diving board","dog","dolphin","donut","door","dragon","dresser"]
model = tf.keras.models.load_model("saved models/"+modelfilename)

engine = pyttsx3.init()
scale = 0
class Paint(object):




    def __init__(self):
        global scale



        self.root = Tk()
        scale =1920/ self.root.winfo_screenwidth()
        print(scale)






        self.eraser_button = Button(self.root, text='erase', command=self.use_eraser ,height = 2, width = 30)
        self.save_button = Button(self.root,text="save", command=self.save, height=2, width=30)
        self.save_button.grid(row=0, column=1)
        self.eraser_button.grid(row=0, column=2)




        self.c = Canvas(self.root, bg='white', width=800/scale, height=800/scale)
        self.c.grid(row=2, columnspan=5)
        self.label1 = Label(self.root, text="", bg="white", height=1, width=60, font=("Courier", 20))
        self.label1.grid(row=4, columnspan=5)

        threading.Thread(target=lambda : self.save()).start()

        self.setup()
        self.root.mainloop()



    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = "black"
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
    def save(self):
        global oldtext
        self.c.postscript(file="drawnimage.eps")
        img = Image.open("drawnimage.eps")
        img=img.resize((28,28))
        img=PIL.ImageOps.invert(img)
        img=img.convert('L')
        imgA=np.asarray(img)
        imgA=imgA.reshape(28,28,1).astype('float32')
        imgA/=255.0



        arr = model.predict(imgA[None,:,:,:])[0]
        indices =  arr.argsort()[-3:][::-1]
        text=""
        if(arr[indices[0]]>0.25):


         text="I see "+labellist[indices[0]]

        else:
            if(arr[indices[0]]>0.16):
               text="I am not sure what that is."

        if not oldtext==text:

            self.label1.config(text=text)
            engine.say(text)
            engine.runAndWait()
            oldtext=text
        for i in indices:
            print(labellist[i] ,str(int(arr[i]*100))+"%",end=",")
        print("----------")

       # plt.figure()
       # plt.imshow(imgA)
       # plt.colorbar()
       # plt.gray()
       # plt.grid(False)
       # plt.show()
        threading.Timer(0.5, lambda: self.save()).start()



    def use_brush(self):
        self.activate_button(self.brush_button)



    def use_eraser(self):
        self.c.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = 30
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y


    def reset(self, event):
        self.old_x, self.old_y = None, None
        #threading.Thread(target=self.save()).start()









if __name__ == '__main__':
    Paint()