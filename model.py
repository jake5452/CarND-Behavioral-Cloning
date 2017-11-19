import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

INPUT_SHAPE = (160, 320, 3)
BATCH_SIZE = 128
EPOCHS = 5
IMG_DATA_PATH = 'data/IMG/'

def apply_random_data_augmentation(img, steering):
    # Add random brightness change
    img = apply_random_brightness_changes(img)

    # Add random translation
    # img, steering = apply_random_translation(img, steering)

    # Add random flip
    img, steering = apply_random_flip(img, steering)

    return img, steering

def subsample_low_angle_data(data):
    """
    Subsample low angle data
    """
    new_data_log = []
    for line in data_log:
        steering = float(line[3])
        if (abs(steering) < 0.05):
            random_drop = np.random.randint(10)
            if (random_drop < 1):
                new_data_log.append(line)
        else:
            new_data_log.append(line)

    return new_data_log

def apply_random_brightness_changes(image):
    """
    Randomly change brightness of the image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 0.25 + 1.0*np.random.rand()
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_random_translation(img, steering):
    """
    Randomly add translation to the images
    """
    # Compute translation to be applied
    trans_range_in_pixels = 40
    tr_x = trans_range_in_pixels * 2 * (np.random.uniform() - 0.5)
    
    # # Change steering angle according to the range change
    steering_change_per_pxiels = 0.005
    steering_transformed = steering + tr_x * steering_change_per_pxiels
    transform = np.float32([[1, 0, tr_x], [0, 1, 0]])
    image_translated = cv2.warpAffine(img, transform, (320,160))
    return image_translated, steering_transformed

def flip_data(img, angle):
    """
    Return horizontally flipped data
    """
    return cv2.flip(img, 1), -angle

def apply_random_flip(img, steering):
    """
    Randomly add flip to the images
    """
    random_flip = np.random.randint(2)
    if (random_flip == 0):
        img, steering = flip_data(img, steering)

    return img, steering

def read_img(source_path, image_data_path):
    filename = source_path.split('/')[-1]
    current_path = image_data_path + filename
    img = cv2.imread(current_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_data_from_log(data_log, image_data_path):
    """
    Generate data
    """
    images = []
    steering = []
    for i in range(len(data_log)):
        # Read for center image 
        line = data_log[i]
        source_path = line[0]
        
        img = read_img(source_path, image_data_path)
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
        images.append(img)
        steering.append(float(line[3]))
    return np.array(images), np.array(steering)

def training_data_genator(data_log, image_data_path, batch_size=128):
    """
    Training data generator
    """
    data_log = shuffle(data_log)
    X,y = ([],[])
    while True:
        for line in data_log:
            correction = 0.20

            # Read for center image 
            source_path = line[0]
            img = read_img(source_path, image_data_path)
            steering = float(line[3])
            img, steering = apply_random_data_augmentation(img, steering)
            X.append(img)
            y.append(steering)

            # Read for left image
            source_path = line[1]
            img_left = read_img(source_path, image_data_path)
            steering_left = steering + correction
            img_left, steering_left = apply_random_data_augmentation(img_left, steering_left)
            X.append(img_left)
            y.append(steering_left)

            # Read for right image
            source_path = line[2]
            img_right = read_img(source_path, image_data_path)
            steering_right = steering - correction
            img_right, steering_right = apply_random_data_augmentation(img_right, steering_right)
            X.append(img_right)
            y.append(steering_right)

            if len(X) > batch_size:
                # Yield the generated bach and reshuffle data
                X = np.resize(np.array(X), (batch_size, img.shape[0], img.shape[1], img.shape[2]))
                y = np.resize(np.array(y), batch_size)

                yield (X, y)
                X, y = ([],[])
                # Shuffle for cross validation
                data_log = shuffle(data_log)

def collect_data_log_from_csv(csv_path):
    """
    Collect data log from csv
    """
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        i = 0
        for line in reader:
            if (i > 0):
                lines.append(line)
            else:
                i += 1
    return lines

def nvidia_model(input_shape):
    """
    Construct nvidia model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100,  activation="relu"))
    model.add(Dense(50,  activation="relu"))
    model.add(Dense(10,  activation="relu"))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=1e-3))

    return model

def visualize_data_distribution(data):
    num_bins = 30
    hist, bins = np.histogram(data, num_bins)
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.7 * (bins[1] - bins[0])
    plt.bar(center, hist, align='center', width=width)
    plt.show()

# Main code

csv_path = 'data/driving_log.csv'
data_log = collect_data_log_from_csv(csv_path)

## Training and Validation Data
data_log = shuffle(data_log) 
_, angles = generate_data_from_log(data_log, IMG_DATA_PATH)
visualize_data_distribution(angles.astype(np.float))

data_log = subsample_low_angle_data(data_log)
images, angles = generate_data_from_log(data_log, IMG_DATA_PATH)
visualize_data_distribution(angles.astype(np.float))

# plt.subplot(1,2,1)
# plt.axis("off")
# plt.imshow(images[0])

# transformed_image = apply_random_brightness_changes(images[0])

# plt.subplot(1,2,2)
# plt.imshow(transformed_image)
# plt.axis("off")
# plt.show()

# fig, axis_arr = plt.subplots(1,2)
# axis_arr[0].imshow(images[0])
# axis_arr[0].axis("off")
# axis_arr[0].set_title("Before Trnasforming/Angle: {0:.5f}".format(angles[0]))
# transformed_img, transformed_steering = apply_random_translation(images[0], angles[0])

# axis_arr[1].imshow(transformed_img)
# axis_arr[1].axis("off")
# axis_arr[1].set_title("Before Trnasforming/Angle: {0:.5f}".format(transformed_steering))
# plt.show()

# fig, axis_arr = plt.subplots(1,2)
# axis_arr[0].imshow(images[0])
# axis_arr[0].axis("off")
# axis_arr[0].set_title("Before Flipping")
# transformed_img, transformed_steering = apply_random_flip(images[0], angles[0])

# axis_arr[1].imshow(transformed_img)
# axis_arr[1].axis("off")
# axis_arr[1].set_title("After Flipping")
# plt.show()

# #80/10/10 split in ttraining, validation, and testing data
# data_log = shuffle(data_log)  # Randomize data set creation
# training_count = int(0.8 * len(data_log))
# validation_count = int(0.1 * len(data_log))
# training_data_log = data_log[:training_count]
# validation_data_log = data_log[training_count:training_count+validation_count]
# test_data_log = data_log[training_count+validation_count:]

# train_data_gen = training_data_genator(training_data_log, IMG_DATA_PATH)
# valid_img, valid_steering = generate_data_from_log(validation_data_log, IMG_DATA_PATH)
# test_img, test_steering = generate_data_from_log(test_data_log, IMG_DATA_PATH)
# print(len(data_log))
# print(len(valid_img))
# print(len(test_img))

# model = nvidia_model(INPUT_SHAPE)
# model.summary()
# model.fit_generator(train_data_gen, 
#                     samples_per_epoch=int(training_count / BATCH_SIZE) * BATCH_SIZE, 
#                     nb_epoch=EPOCHS, 
#                     validation_data=[valid_img, valid_steering])

# # Evaluate on test data
# test_loss = model.evaluate(test_img, test_steering, batch_size=128)
# print('Test Loss: ', test_loss)    # Loss on test set

# model.save('model.h5')