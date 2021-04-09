import urllib.request
import os
import sys
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

download = False


modelfilename="modelallwords"
image_size=28
num_classes=100
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
labellist=["The Eiffel Tower" ,"The Great Wall of China" ,"The Mona Lisa" ,"aircraft carrier" ,"airplane" ,"alarm clock" ,"ambulance" ,"angel" ,"animal migration" ,"ant" ,"anvil" ,"apple" ,"arm" ,"asparagus" ,"axe" ,"backpack" ,"banana" ,"bandage" ,"barn" ,"baseball bat" ,"baseball" ,"basket" ,"basketball" ,"bat" ,"bathtub" ,"beach" ,"bear" ,"beard" ,"bed" ,"bee" ,"belt" ,"bench" ,"bicycle" ,"binoculars" ,"bird" ,"birthday cake" ,"blackberry" ,"blueberry" ,"book" ,"boomerang" ,"bottlecap" ,"bowtie" ,"bracelet" ,"brain" ,"bread" ,"bridge" ,"broccoli" ,"broom" ,"bucket" ,"bulldozer" ,"bus" ,"bush" ,"butterfly" ,"cactus" ,"cake" ,"calculator" ,"calendar" ,"camel" ,"camera" ,"camouflage" ,"campfire" ,"candle" ,"cannon" ,"canoe" ,"car" ,"carrot" ,"castle" ,"cat" ,"ceiling fan" ,"cell phone" ,"cello" ,"chair" ,"chandelier" ,"church" ,"circle" ,"clarinet" ,"clock" ,"cloud" ,"coffee cup" ,"compass" ,"computer" ,"cookie" ,"cooler" ,"couch" ,"cow" ,"crab" ,"crayon" ,"crocodile" ,"crown" ,"cruise ship" ,"cup" ,"diamond" ,"dishwasher" ,"diving board" ,"dog" ,"dolphin" ,"donut" ,"door" ,"dragon" ,"dresser" ,"drill" ,"drums" ,"duck" ,"dumbbell" ,"ear" ,"elbow" ,"elephant" ,"envelope" ,"eraser" ,"eye" ,"eyeglasses" ,"face" ,"fan" ,"feather" ,"fence" ,"finger" ,"fire hydrant" ,"fireplace" ,"firetruck" ,"fish" ,"flamingo" ,"flashlight" ,"flip flops" ,"floor lamp" ,"flower" ,"flying saucer" ,"foot" ,"fork" ,"frog" ,"frying pan" ,"garden hose" ,"garden" ,"giraffe" ,"goatee" ,"golf club" ,"grapes" ,"grass" ,"guitar" ,"hamburger" ,"hammer" ,"hand" ,"harp" ,"hat" ,"headphones" ,"hedgehog" ,"helicopter" ,"helmet" ,"hexagon" ,"hockey puck" ,"hockey stick" ,"horse" ,"hospital" ,"hot air balloon" ,"hot dog" ,"hot tub" ,"hourglass" ,"house plant" ,"house" ,"hurricane" ,"ice cream" ,"jacket" ,"jail" ,"kangaroo" ,"key" ,"keyboard" ,"knee" ,"knife" ,"ladder" ,"lantern" ,"laptop" ,"leaf" ,"leg" ,"light bulb" ,"lighter" ,"lighthouse" ,"lightning" ,"line" ,"lion" ,"lipstick" ,"lobster" ,"lollipop" ,"mailbox" ,"map" ,"marker" ,"matches" ,"megaphone" ,"mermaid" ,"microphone" ,"microwave" ,"monkey" ,"moon" ,"mosquito" ,"motorbike" ,"mountain" ,"mouse" ,"moustache" ,"mouth" ,"mug" ,"mushroom" ,"nail" ,"necklace" ,"nose" ,"ocean" ,"octagon" ,"octopus" ,"onion" ,"oven" ,"owl" ,"paint can" ,"paintbrush" ,"palm tree" ,"panda" ,"pants" ,"paper clip" ,"parachute" ,"parrot" ,"passport" ,"peanut" ,"pear" ,"peas" ,"pencil" ,"penguin" ,"piano" ,"pickup truck" ,"picture frame" ,"pig" ,"pillow" ,"pineapple" ,"pizza" ,"pliers" ,"police car" ,"pond" ,"pool" ,"popsicle" ,"postcard" ,"potato" ,"power outlet" ,"purse" ,"rabbit" ,"raccoon" ,"radio" ,"rain" ,"rainbow" ,"rake" ,"remote control" ,"rhinoceros" ,"rifle" ,"river" ,"roller coaster" ,"rollerskates" ,"sailboat" ,"sandwich" ,"saw" ,"saxophone" ,"school bus" ,"scissors" ,"scorpion" ,"screwdriver" ,"sea turtle" ,"see saw" ,"shark" ,"sheep" ,"shoe" ,"shorts" ,"shovel" ,"sink" ,"skateboard" ,"skull" ,"skyscraper" ,"sleeping bag" ,"smiley face" ,"snail" ,"snake" ,"snorkel" ,"snowflake" ,"snowman" ,"soccer ball" ,"sock" ,"speedboat" ,"spider" ,"spoon" ,"spreadsheet" ,"square" ,"squiggle" ,"squirrel" ,"stairs" ,"star" ,"steak" ,"stereo" ,"stethoscope" ,"stitches" ,"stop sign" ,"stove" ,"strawberry" ,"streetlight" ,"string bean" ,"submarine" ,"suitcase" ,"sun" ,"swan" ,"sweater" ,"swing set" ,"sword" ,"syringe" ,"t-shirt" ,"table" ,"teapot" ,"teddy-bear" ,"telephone" ,"television" ,"tennis racquet" ,"tent" ,"tiger" ,"toaster" ,"toe" ,"toilet" ,"tooth" ,"toothbrush" ,"toothpaste" ,"tornado" ,"tractor" ,"traffic light" ,"train" ,"tree" ,"triangle" ,"trombone" ,"truck" ,"trumpet" ,"umbrella" ,"underwear" ,"van" ,"vase" ,"violin" ,"washing machine" ,"watermelon" ,"waterslide" ,"whale" ,"wheel" ,"windmill" ,"wine bottle" ,"wine glass" ,"wristwatch" ,"yoga" ,"zebra" ,"zigzag" ]
print(len(labellist))
base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'





def load_data(root, vfold_ratio=0.5, max_items_per_class= 5000 ):


    #initialize variables
    global download
    x = np.empty([0, 784])
    y = np.empty([0])


    #load a subset of the data to memory
    for idx,i in enumerate(labellist):
        file=root+"/"+i+".npy"
        if(os.path.exists(file) == False):

            if not download:
             shoulddownload = input("Some training files doesnt exist do you want to download them? (y/n)")
             if(shoulddownload.lower()=="y"):
               download=True
             elif(shoulddownload.lower()=="n"):
                sys.exit("You canceled the training dataset download")
            print("Downloading: "+base + i.replace(" ", "%20")+".npy")
            urllib.request.urlretrieve(base + i.replace(" ", "%20") + '.npy', root+"/" + i + '.npy')

        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)



    #separate into training and testing
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]
    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))
    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test= load_data("data")
if(download==True):
    print("Download complete")


x_train = x_train.reshape(x_train.shape[0], image_size, image_size,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size,1).astype('float32')
print(x_train.shape)
x_train /= 255.0
x_test /= 255.0


model = keras.Sequential()
model.add(layers.Convolution2D(16, (3, 3),
                        padding='same',
                        input_shape=x_train.shape[1:], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2,2)))
model.add(layers.Convolution2D(128, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(labellist), activation='softmax'))
# Train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],)

#fit the model
model.fit(x_train, y_train, validation_split=0.1, batch_size = 384, verbose=2, epochs=20)
tf.saved_model.save(model, "saved models/"+modelfilename)
#evaluate on unseen data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))
for i in range(32):
 predictions = model.predict(x_test)
 plt.figure()
 plt.imshow(x_test[i])
 plt.colorbar()
 plt.gray()
 plt.grid(False)
 plt.show()

 print(labellist[np.argmax(predictions[i])])
