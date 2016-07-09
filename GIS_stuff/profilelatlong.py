"""Script to use call the pixeltolatlng_better function to profile it and save the profile"""

import numpy as np

import gmapshelpers as ghelpers

# for i in range(100):
#	ltng_final = ghelpers.pixelToLatlng_better((i,i),i%20)
#	print i

px = np.arange(100)
py = np.arange(100)
z = np.arange(100) % 20
ltlng_final = ghelpers.pixelToLatlng_better((px, py), z)

print(ltlng_final)
