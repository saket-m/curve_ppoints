import poha
import pista
import utils
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

manual_dir = ''         ## dir where manually generated images are stored
manual = pista.read_images_into_dict(manual_dir)

gen_dir = ''            # dir where semantically segmented images are stored
gen = pista.read_images_into_dict(gen_dir)

## calculate mse
mse = 0
for key in manual.keys():
    mse += mean_squared_error(manual[key] > 0, gen[key] > 0) / len(manual.keys())
print('MSE =', mse)

## calculate meanIOU

