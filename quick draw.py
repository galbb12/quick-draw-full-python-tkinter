from tkinter import *
import PIL.ImageOps
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

modelfilename="model1"
labellist=["The Eiffel Tower","The Great Wall of China","The Mona Lisa","aircraft carrier","airplane","alarm clock","ambulance","angel","animal migration","ant","anvil","apple","arm","asparagus","axe","backpack","banana","bandage","barn","baseball bat","baseball","basket","basketball","bat","bathtub","beach","bear","beard","bed","bee","belt","bench","bicycle","binoculars","bird","birthday cake","blackberry","blueberry","book","boomerang","bottlecap","bowtie","bracelet","brain","bread","bridge","broccoli","broom","bucket","bulldozer","bus","bush","butterfly","cactus","cake","calculator","calendar","camel","camera","camouflage","campfire","candle","cannon","canoe","car","carrot","castle","cat","ceiling fan","cell phone","cello","chair","chandelier","church","circle","clarinet","clock","cloud","coffee cup","compass","computer","cookie","cooler","couch","cow","crab","crayon","crocodile","crown","cruise ship","cup","diamond","dishwasher","diving board","dog","dolphin","donut","door","dragon","dresser"]
model = tf.keras.models.load_model("saved models/"+modelfilename)
class Paint(object):




    def __init__(self):
        self.root = Tk()




        self.eraser_button = Button(self.root, text='erase', command=self.use_eraser ,height = 2, width = 30)
        self.save_button = Button(self.root, text='save', command=self.save, height=2, width=30)
        self.save_button.grid(row=0, column=1)
        self.eraser_button.grid(row=0, column=2)


        self.c = Canvas(self.root, bg='white', width=840, height=840)
        self.c.grid(row=1, columnspan=5)

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
        self.c.postscript(file="drawnimage.eps")
        img = Image.open("drawnimage.eps")
        img=img.resize((28,28))
        img=PIL.ImageOps.invert(img)
        img.save("drawnimage.png", "PNG")

        img = plt.imread("drawnimage.png")

        img.resize(28,28,1)




        print(labellist[np.argmax(model.predict(img[None,:,:]))])



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
        self.line_width = 20
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()