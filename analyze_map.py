"""
analyze_map.py: Analyze a set of maps for dominant colors
and return their pixel counts as a function of time (assuming
different images represent different times).

Copyright: Marco Tompitak 2016

Usage:

python analyze_map.py <colorspace binning radius> <folder>
          [unique color]

Example:

python analyze_map.py 10 Test '#00489C'

Description:

This script runs the Dynamic Color Binning algorithm on a set
of images that represent maps. (It can be run on any set of images
but is designed for images with a small number of colors.)

The algorithm requires a radius in L*a*b colorspace to do its
binning. Heuristically, colors that are closer together than
this radius in L*a*b colorspace are considered identical.

The script takes a directory as input and assumes that all files
in this directory are images to be analyzed. The user should take
care to set this up properly.

Optionally one can provide a "unique color", where unique means
a color that is not close to any that is represented in the images.
This color will be used for the text overlays in the color palette.
"""

import os
import re
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
from matplotlib.colors import hex2color, rgb2hex

import cv2
import dynamiccolorbinning as dcb
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor

filterstrength = 0


# SCRIPT START
# Read in command line arguments.

# Bin size in Lab color space.
r = int(sys.argv[1])

# Folder to analyze.
folder = sys.argv[2]

# Color for text overlay on palette image.
if(len(sys.argv) > 3):
    uniquehex = sys.argv[3]
    uniquecolor = [int(255*i) for i in hex2color(uniquehex)]
else:
    uniquecolor = [0, 0, 255]


# Create new SetOfMaps object.
Maps = dcb.SetOfMaps()

# Loop through files in folder
for fn in os.listdir(folder):
    filename = './' + folder + '/' + fn
    if os.path.isfile(filename):
        # Find date in filename. This script assumes there is at least
        # a four-digit year in the name! Otherwise the whole filename
        # is used.
        date = re.search("([0-9]{4}-[0-9]{2}-)", filename)
        if (date is None):
            date = re.search("([0-9]{4})", filename).group(0) + "-01"
        else:
            date = date.group(0)[:-1]

        # Load the image with OpenCV.
        print("Loading image... " + filename)
        img = cv2.imread(filename)

        # If desired, a denoising filter can be applied.
        if (filterstrength > 0.0):
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Create a Map object from the image, apply the Dynamic Color
        # Binning algorithm and add to our SetOfMaps, labeled with the
        # date extracted from the filename.
        geo_map = dcb.Map(img)
        geo_map.run_dynamic_binning()
        Maps.add_map(geo_map, date)


# Populate the SetOfMaps' dataframe and clean up.
print("Analyzing set of images...")
Maps.update_dataframe()
Maps.bin_dataframe(r)
Maps.filter_dataframe(4000)


# Convert headers to hex color codes and store as csv.
print("Saving output to " + folder + ".csv")
Maps.hexheader_dataframe().to_csv(folder+'.csv')


# Generate and save palette of abundant colors.
print("Saving palette to " + folder + "_palette.png")
paletteimg = Maps.palette(uniquecolor)
cv2.imwrite(folder+'_palette.png', paletteimg)

print("Done.")
