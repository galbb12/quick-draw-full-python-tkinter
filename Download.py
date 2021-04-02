
import urllib.request
import os


labellist=["The Eiffel Tower","The Great Wall of China","The Mona Lisa","aircraft carrier","airplane","alarm clock","ambulance","angel","animal migration","ant","anvil","apple","arm","asparagus","axe","backpack","banana","bandage","barn","baseball bat","baseball","basket","basketball","bat","bathtub","beach","bear","beard","bed","bee","belt","bench","bicycle","binoculars","bird","birthday cake","blackberry","blueberry","book","boomerang","bottlecap","bowtie","bracelet","brain","bread","bridge","broccoli","broom","bucket","bulldozer","bus","bush","butterfly","cactus","cake","calculator","calendar","camel","camera","camouflage","campfire","candle","cannon","canoe","car","carrot","castle","cat","ceiling fan","cell phone","cello","chair","chandelier","church","circle","clarinet","clock","cloud","coffee cup","compass","computer","cookie","cooler","couch","cow","crab","crayon","crocodile","crown","cruise ship","cup","diamond","dishwasher","diving board","dog","dolphin","donut","door","dragon","dresser"]
base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

def download():

    for c in labellist:
        path = base + c.replace(" ","%20") + '.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/' + c + '.npy')
download()

