from tkinter import *
import pyttsx3
import PIL.ImageOps
from PIL import Image
import numpy as np
from PIL import EpsImagePlugin
import tensorflow as tf
# import matplotlib.pyplot as plt
import threading
import random
import time

oldtext = ""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
EpsImagePlugin.gs_windows_binary = r'bin\gswin64c'

modelfilename = "modelallwords"
labellist = ["The Eiffel Tower", "The Great Wall of China", "The Mona Lisa", "aircraft carrier", "airplane",
             "alarm clock", "ambulance", "angel", "animal migration", "ant", "anvil", "apple", "arm", "asparagus",
             "axe", "backpack", "banana", "bandage", "barn", "baseball bat", "baseball", "basket", "basketball", "bat",
             "bathtub", "beach", "bear", "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird",
             "birthday cake", "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", "bracelet",
             "brain", "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer", "bus", "bush", "butterfly",
             "cactus", "cake", "calculator", "calendar", "camel", "camera", "camouflage", "campfire", "candle",
             "cannon", "canoe", "car", "carrot", "castle", "cat", "ceiling fan", "cell phone", "cello", "chair",
             "chandelier", "church", "circle", "clarinet", "clock", "cloud", "coffee cup", "compass", "computer",
             "cookie", "cooler", "couch", "cow", "crab", "crayon", "crocodile", "crown", "cruise ship", "cup",
             "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut", "door", "dragon", "dresser", "drill",
             "drums", "duck", "dumbbell", "ear", "elbow", "elephant", "envelope", "eraser", "eye", "eyeglasses", "face",
             "fan", "feather", "fence", "finger", "fire hydrant", "fireplace", "firetruck", "fish", "flamingo",
             "flashlight", "flip flops", "floor lamp", "flower", "flying saucer", "foot", "fork", "frog", "frying pan",
             "garden hose", "garden", "giraffe", "goatee", "golf club", "grapes", "grass", "guitar", "hamburger",
             "hammer", "hand", "harp", "hat", "headphones", "hedgehog", "helicopter", "helmet", "hexagon",
             "hockey puck", "hockey stick", "horse", "hospital", "hot air balloon", "hot dog", "hot tub", "hourglass",
             "house plant", "house", "hurricane", "ice cream", "jacket", "jail", "kangaroo", "key", "keyboard", "knee",
             "knife", "ladder", "lantern", "laptop", "leaf", "leg", "light bulb", "lighter", "lighthouse", "lightning",
             "line", "lion", "lipstick", "lobster", "lollipop", "mailbox", "map", "marker", "matches", "megaphone",
             "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike", "mountain", "mouse",
             "moustache", "mouth", "mug", "mushroom", "nail", "necklace", "nose", "ocean", "octagon", "octopus",
             "onion", "oven", "owl", "paint can", "paintbrush", "palm tree", "panda", "pants", "paper clip",
             "parachute", "parrot", "passport", "peanut", "pear", "peas", "pencil", "penguin", "piano", "pickup truck",
             "picture frame", "pig", "pillow", "pineapple", "pizza", "pliers", "police car", "pond", "pool", "popsicle",
             "postcard", "potato", "power outlet", "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "rake",
             "remote control", "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", "sailboat", "sandwich",
             "saw", "saxophone", "school bus", "scissors", "scorpion", "screwdriver", "sea turtle", "see saw", "shark",
             "sheep", "shoe", "shorts", "shovel", "sink", "skateboard", "skull", "skyscraper", "sleeping bag",
             "smiley face", "snail", "snake", "snorkel", "snowflake", "snowman", "soccer ball", "sock", "speedboat",
             "spider", "spoon", "spreadsheet", "square", "squiggle", "squirrel", "stairs", "star", "steak", "stereo",
             "stethoscope", "stitches", "stop sign", "stove", "strawberry", "streetlight", "string bean", "submarine",
             "suitcase", "sun", "swan", "sweater", "swing set", "sword", "syringe", "t-shirt", "table", "teapot",
             "teddy-bear", "telephone", "television", "tennis racquet", "tent", "tiger", "toaster", "toe", "toilet",
             "tooth", "toothbrush", "toothpaste", "tornado", "tractor", "traffic light", "train", "tree", "triangle",
             "trombone", "truck", "trumpet", "umbrella", "underwear", "van", "vase", "violin", "washing machine",
             "watermelon", "waterslide", "whale", "wheel", "windmill", "wine bottle", "wine glass", "wristwatch",
             "yoga", "zebra", "zigzag"]
print(len(labellist))
model = tf.keras.models.load_model("saved models/" + modelfilename)
randomword = ""
engine = pyttsx3.init()
scale = 0


class Paint(object):

    def __init__(self):
        global scale

        self.root = Tk()
        self.root.title("Quick draw by: Gal Bareket")
        scale = 1080 / self.root.winfo_screenheight()
        print(scale)

        self.eraser_button = Button(self.root, text='erase', command=self.use_eraser, height=int(2/scale), width=int(30/scale))

        self.eraser_button.grid(row=0, column=1)

        self.skip_button = Button(self.root, text='skip', command=self.pickword, height=int(2/scale), width=int(30/scale))

        self.skip_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=int(896 / scale), height=int(896 / scale))
        self.c.grid(row=2, columnspan=5)
        self.label1 = Label(self.root, text="", bg="white", height=int(1/scale), width=int(60/scale), font=("Courier", int(20/scale)))
        self.label1.grid(row=4, columnspan=5)
        self.label2 = Label(self.root, text="", bg="white", height=int(1/scale), width=int(35/scale), font=("Courier", int(15/scale)), anchor="w")
        self.label2.grid(row=0, column=2)

        threading.Thread(target=lambda: self.save()).start()

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = "black"
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<Button-1>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.pickword()

    def pickword(self):
        global randomword
        randomword = labellist[random.randint(0, len(labellist)-1)]
        self.label2.configure(text="Draw: " + randomword)
        self.use_eraser()

    def save(self):
        global oldtext
        won = False
        self.c.postscript(file="drawnimage.eps")
        img = Image.open("drawnimage.eps")
        img = img.resize((28, 28))
        img = PIL.ImageOps.invert(img)
        img = img.convert('L')
        imgA = np.asarray(img)
        imgA = imgA.reshape(28, 28, 1).astype('float32')
        imgA /= 255.0

        arr = model.predict(imgA[None, :, :, :])[0]
        indices = arr.argsort()[-3:][::-1]
        predictionlist = []
        for i in arr.argsort():
            if (arr[i] > 0.10):
                predictionlist.append(labellist[i])

        text = ""
        if (randomword in predictionlist):
            text = "Oh i know it's " + randomword
            won = True
        elif (arr[indices[0]] > 0.10):
            for i in range(2):
                if (arr[indices[i]] > 0.10):
                    if (i == 0):
                        text = "I see " + labellist[indices[0]]
                    else:
                        text += ", " + labellist[indices[i]]




        else:
            if (arr[indices[0]] > 0.5):
                text = "I am not sure what that is."

        for i in indices:
            print(labellist[i], str(int(arr[i] * 100)) + "%", end=",")
        print("----------")

        if not oldtext == text:
            self.label1.config(text=text)
            engine.say(text.replace(",", " or "))
            engine.runAndWait()
            oldtext = text
        if (randomword in predictionlist):
            time.sleep(2)
            self.pickword()
            time.sleep(2)

        # plt.figure()
        # plt.imshow(imgA)
        # plt.colorbar()
        # plt.gray()
        # plt.grid(False)
        # plt.show()

        threading.Timer(0.25, lambda: self.save()).start()

    def use_eraser(self):
        self.c.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):

        self.line_width = 40 / scale
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        else:
            self.c.create_line(event.x, event.y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        # threading.Thread(target=self.save()).start()


if __name__ == '__main__':
    Paint()
