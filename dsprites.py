import time
import random
import imutil
import numpy as np

NUM_FACTORS = 4
imgs = None

def init():
    global latents_classes
    global imgs
    start_time = time.time()
    print('Loading dsprites...')
    with np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes') as npz:
        #metadata = npz['metadata']
        imgs = npz['imgs']
        latents_classes = npz['latents_classes']
        #latents_values = npz['latents_values']
    print('Finished loading dsprites in {:.2f} sec'.format(time.time() - start_time))


# NOTE: Following Higgins et al 2016 we DO NOT use shape as a latent variable
#   Shape (discrete, 3 values)
# There are 4 ground truth factors:
#   Scale (discrete, 6 values)
#   Orientation (discrete, 40 values)
#   Pos X (discrete, 32 values)
#   Pos Y (discrete, 32 values)
def generate_image_discrete(factors):
    if imgs is None:
        raise Exception('dsprites was not initialized, call dsprites.init()')
    # scale, orientation, x, y = factors
    shape = 0
    idx =  factors[3]
    idx += factors[2] * 32
    idx += factors[1] * 32*32
    idx += factors[0] * 32*32*40
    idx += shape * 32*32*40*6
    # Hack: subsample down to 32x32 for the architecture
    return imgs[idx, ::2, ::2]


def generate_image_continuous(factors):
    discrete = np.zeros(shape=(NUM_FACTORS,), dtype=int)
    discrete[3] = int(factors[3] * 32)
    discrete[2] = int(factors[2] * 32)
    discrete[1] = int(factors[1] * 40)
    discrete[0] = int(factors[0] * 6)
    #discrete[0] = int(factors[0] * 3)
    return generate_image_discrete(discrete)


def get_example():
    #shape = random.randint(0, 2)
    scale = random.randint(0, 5)
    orientation = random.randint(0, 39)
    x = random.randint(0, 31)
    y = random.randint(0, 31)
    factors = [scale, orientation, x, y]
    return (generate_image_discrete(factors), normalize_factors(factors))


def normalize_factors(factors):
    #factors[0] /= 3.
    factors[0] /= 6.
    factors[1] /= 40.
    factors[2] /= 32.
    factors[3] /= 32.
    return factors


def get_batch(batch_size=32):
    x = np.zeros((batch_size, 1, 32, 32))
    target = np.zeros((batch_size, NUM_FACTORS))
    for i in range(batch_size):
        x[i], target[i] = get_example()
    return x, target


# For use with higgins_metric
def simulator(factor_batch):
    batch_size = len(factor_batch)
    images = []
    for i in range(batch_size):
        images.append(generate_image_continuous(factor_batch[i]))
    return np.array(images)

