import pattern
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator
import matplotlib.pyplot as plt


ch = Checker(10,2)
cir = Circle(100, 15, (50, 30))
sp = Spectrum(100)

cir.show()
sp.show()
ch.show()

## generator
file_path = "./exercise_data"
label_path = "./Labels.json"
batch_size = 8
image_size = (32,32,3)

gen = ImageGenerator(file_path, label_path, batch_size, image_size, mirroring=True, rotation=True)
gen.show()