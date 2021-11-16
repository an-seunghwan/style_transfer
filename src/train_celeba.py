#%%
import tensorflow as tf
import tensorflow.keras as K
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
import numpy as np
from tensorflow.keras.applications import vgg19
from IPython.display import Image, display
from tqdm import tqdm
import os

os.chdir('D:\style_transfer')
#%%
PARAMS = {
    # Weights of the different loss components
    'total_variation_weight': 1e-6,
    'style_weight': 1e-6,
    'content_weight': 2.5e-8,
    'img_nrows': 400,
    'channels': 3,
    'iterations': 4000,
    'initial_learning_rate': 100.0, 
    'decay_steps': 100, 
    'decay_rate': 0.96
}
#%%
base_image_path = K.utils.get_file(
    "paris.jpg", "https://i.imgur.com/F28w3Ac.jpg"
)
style_reference_image_path = K.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"

# Dimensions of the generated picture.
width, height = K.preprocessing.image.load_img(base_image_path).size
print(width)
print(height)
PARAMS['img_ncols'] = int(width * PARAMS['img_nrows'] / height)
#%%
display(Image(base_image_path))
display(Image(style_reference_image_path))
#%%
'''Util function to open, resize and format pictures into appropriate tensors'''
def preprocess_image(image_path):
    img = K.preprocessing.image.load_img(
        image_path, target_size=(PARAMS['img_nrows'], PARAMS['img_ncols'])
    )
    img = K.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

'''Util function to convert a tensor into a valid image'''
def deprocess_image(x):
    x = x.reshape((PARAMS['img_nrows'], PARAMS['img_ncols'], 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
#%%
base_image = preprocess_image(base_image_path)
base_image.shape
style_reference_image = preprocess_image(style_reference_image_path)
style_reference_image.shape
#%%
'''Build a VGG19 model loaded with pre-trained ImageNet weights'''
model = vgg19.VGG19(weights="imagenet", include_top=False)

'''Get the symbolic outputs of each "key" layer (we gave them unique names).'''
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = ["block5_conv2"]

outputs_dict = {k : i for k, i in outputs_dict.items() if k in style_layer_names + content_layer_name}

'''Set up a model that returns the activation values for every layer in VGG19 (as a dict).'''
feature_extractor = K.Model(inputs=model.inputs, outputs=outputs_dict)
#%%
@tf.function
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

'''The gram matrix of an image tensor (feature-wise outer product)'''
# only single image
@tf.function
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

@tf.function
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = PARAMS['img_nrows'] * PARAMS['img_ncols']
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (PARAMS['channels'] ** 2) * (size ** 2))

'''designed to keep the generated image locally coherent'''
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
#%%
@tf.function
def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name[0]]
    base_image_features, _, combination_features = tf.split(layer_features, num_or_size_splits=3, axis=0)
    loss = loss + PARAMS['content_weight'] * content_loss(base_image_features, combination_features)
    
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        _, style_reference_features, combination_features = tf.split(layer_features, num_or_size_splits=3, axis=0)
        loss += (PARAMS['style_weight'] / len(style_layer_names)) * style_loss(tf.squeeze(style_reference_features), tf.squeeze(combination_features))

    # Add total variation loss
    loss += PARAMS['total_variation_weight'] * total_variation_loss(combination_image)
    return loss
#%%
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads
#%%
optimizer = K.optimizers.SGD(
    K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=PARAMS['initial_learning_rate'], 
        decay_steps=PARAMS['decay_steps'], 
        decay_rate=PARAMS['decay_rate']
    )
)

step = 0
progress_bar = tqdm(range(PARAMS['iterations']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['iterations']))

# variable to optimize
combination_image = tf.Variable(preprocess_image(base_image_path))

for _ in progress_bar:
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    
    step += 1
    
    progress_bar.set_description('iteration {}/{} | loss {:.3f}'.format(
        step, PARAMS['iterations'], loss
    )) 
    
    if step % 500 == 0:
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_{}.png".format(step)
        K.preprocessing.image.save_img('./assets/' + fname, img)
#%%
display(Image('./assets/' + result_prefix + "_at_iteration_4000.png"))
#%%