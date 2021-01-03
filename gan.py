import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Cropping2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
from tensorflow.keras import backend

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

LATENT_SPACE     = 240
EPOCHS           = 10000
BATCH_SIZE       = 16
SAMPLE_FREQUENCY = 100
SAMPLE_QUANTITY  = 5
NAME             = 'GAN 1.3 - adam (0.0001, 0.00005) - 240_latent'  

# image size: 109 x 89

# set the seed to a consistant value
np.random.seed(1)
tf.random.set_seed(1)

Path(f'results/{NAME}').mkdir(parents=True, exist_ok=True)
# delete any files in the name directory
files = glob.glob(os.path.join(f"results/{NAME}", "**/*"), recursive=True)
files = [f for f in files if os.path.isfile(f)]
for file in files:
    os.remove(file)

Path(f'results/{NAME}/images').mkdir(parents=True, exist_ok=True)
Path(f'results/{NAME}/models').mkdir(parents=True, exist_ok=True)
 
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

def buildGenerator(summary=False):

    i = Input( shape=(LATENT_SPACE, ) )

    x = Dense( LATENT_SPACE)(i)
    x = LeakyReLU(0.2)(x)

    x = BatchNormalization(momentum=0.9)(x)

    x = Reshape( target_shape=(4, 3, 20) )(x)

    x = Dropout(0.4)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose( 1024, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.9)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose( 512, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.9)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose( 256 , 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.9)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose( 128 , 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.9)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose( 3  , 3, padding='same', activation="sigmoid")(x)

    x = Cropping2D( cropping=((10, 9), (4, 3)) )(x)

    model = Model(i, x)
    if (summary):
        print(model.summary())

    return model

def buildDiscriminator(summary=False):

    i = Input( shape=(109, 89, 3) )

    x = Conv2D( filters=256, kernel_size=8, strides=2, padding='same')(i)
    x = LeakyReLU(0.2)(x)

    x = Conv2D( filters=128, kernel_size=8, strides=2, padding='same')(i)
    x = LeakyReLU(0.2)(x)

    x = Conv2D( filters=64, kernel_size=8, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D( filters=32, kernel_size=8, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D( filters=16 , kernel_size=8, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    x = Dense( 1, activation='sigmoid' )(x)

    model = Model(i, x)
    if (summary):
        print(model.summary())

    return model

def loadTrainingImages(source="animal-dataset.npy"):
    images = np.load(source)

    return images

def saveModels(epoch):
    generator.save(f'results/{NAME}/models/{epoch}-generator.h5')
    discriminator.save(f'results/{NAME}/models/{epoch}-discriminator.h5')

def sampleImages(epoch, noise, dim=5):
    rows, cols = dim, dim
    images = generator.predict(noise)  
    plt.figure(dpi=1000)

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(images[idx])
            axs[i, j].axis('off')
            idx += 1
    fig.savefig(f'results/{NAME}/images/{epoch}.png')
    fig.clear()
    plt.close(fig)

# build the generator and discriminator ------------------------------------------------------------------------------------------------------------------------
generator     = buildGenerator(True)
discriminator = buildDiscriminator(True)

discriminator.compile(
    loss='binary_crossentropy',
    optimizer = Adam(lr=0.0001, beta_1=0.5)
)

# build the GAN        -----------------------------------------------------------------------------------------------------------------------------------------
i = Input( shape=(LATENT_SPACE) )
x = generator(i)

discriminator.trainable = False

x = discriminator(x)

gan = Model(i, x)
gan.compile(
    loss='binary_crossentropy',
    optimizer = Adam(lr=0.00005, beta_1=0.5)
)

print(gan.summary())

# load images          -----------------------------------------------------------------------------------------------------------------------------------------
images = loadTrainingImages('../../../Datasets/Celeba Dataset/celeba_min.npy')
print('Training images:', images.shape)

# begin training the GAN -----------------------------------------------------------------------------------------------------------------------------------------
ones  = np.ones(BATCH_SIZE) * 0.9
zeros = np.zeros(BATCH_SIZE)

d_loss = []
g_loss = []

img_noise = np.random.normal(size=(SAMPLE_QUANTITY * SAMPLE_QUANTITY, LATENT_SPACE))

for e in tqdm(range(1, EPOCHS+1)):
    
        # Train the discriminator -----------------------------------------------------------------------------------------------------------------------------------------
    # select a random batch of the training images
    idx = np.random.randint(0, images.shape[0], BATCH_SIZE)
    real_images = images[idx]
    
    # generate an equal amount of fake images with the generator
    noise = np.random.normal(size=(BATCH_SIZE, LATENT_SPACE))
    fake_images = generator.predict(noise)

    # fit the discriminator
    r1 = discriminator.fit(real_images, ones , verbose=0, batch_size=BATCH_SIZE)
    r2 = discriminator.fit(fake_images, zeros, verbose=0, batch_size=BATCH_SIZE)

    # record the discriminator's loss
    d_loss.append((r1.history['loss'][0] + r2.history['loss'][0]) / 2)

        # Train the generator     -----------------------------------------------------------------------------------------------------------------------------------------
    noise = np.random.normal(size=(BATCH_SIZE, LATENT_SPACE))

    r = gan.fit(noise, ones, verbose=0, batch_size=BATCH_SIZE)

    # record the generator's loss
    g_loss.append(r.history['loss'])

        # Sample the generator    -----------------------------------------------------------------------------------------------------------------------------------------
    if (e % SAMPLE_FREQUENCY == 0):
        sampleImages(e, img_noise, dim=SAMPLE_QUANTITY)
        saveModels(e)

# plot the generator and discriminator's losses over time
plt.plot(d_loss, label='discriminator')
plt.legend(loc=1)
plt.savefig(f'results/{NAME}/d_loss.pdf')
plt.savefig(f'results/{NAME}/d_loss.png')
plt.cla()

plt.plot(g_loss, label='generator')
plt.legend(loc=1)
plt.savefig(f'results/{NAME}/g_loss.pdf')
plt.savefig(f'results/{NAME}/g_loss.png')
plt.cla()

plt.plot(d_loss, label='discriminator')
plt.plot(g_loss, label='generator')
plt.legend(loc=1)
plt.savefig(f'results/{NAME}/loss.pdf')
plt.savefig(f'results/{NAME}/loss.png')