"""
dynamiccolorbinning.py: a module that provides functionality
associated with the Dynamic Color Binning algorithm.

Copyright: Marco Tompitak 2016
"""

from ast import literal_eval

import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex

import cv2
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor


class ColorList:
    """
    This class represents a list of colors with their pixel counts as found in
    an associated image. It can be constructed from an OpenCV color histogram
    (which should be a numpy array). The constructor will convert such a
    histogram into the right shape. A ColorList can also be constructed
    from a list using the .from_list() method.
    The actual data structure represented by a ColorList object is an N-by-4
    numpy array. In each row, the first three values are the R, G and B values
    that represent the color. The fourth value is the pixel count.
    """

    def __init__(self, hist=None):
        if (hist is not None):
            self.colorlist = np.zeros([hist.size, 4], 'd', order='F')

            NI, NJ, NK = hist.shape

            # build columns for (i,j,k) tuples using repeat and tile
            self.colorlist[:, 0] = np.repeat(range(NI), NJ*NK)
            self.colorlist[:, 1] = np.tile(np.repeat(range(NJ), NK), NI)
            self.colorlist[:, 2] = np.tile(range(NK), NI*NJ)
            self.colorlist[:, 3] = hist.flatten()
            reduced_colorlist = self.colorlist[self.colorlist[:, 3] > 0.0]
            self.colorlist = \
                reduced_colorlist[reduced_colorlist[:, 3].argsort()[
                    ::-1]].astype(int).tolist()

    @classmethod
    def from_list(cls, src):
        """
        Class method to construct a ColorList object from a list.
        """
        clist = cls()
        clist.colorlist = src
        return clist

    def __getitem__(self, key):
        return self.colorlist[key]

    def dynamic_binning(self, radius):
        """
        This function applies the dynamic color binning algorithm to the
        ColorList. This algorithm sorts the colors by pixel count. Selecting
        the most prominent color, it searches the rest of the list for similar
        (i.e. within a distance <radius> in Lab color space) colors and adds
        the pixel counts of those colors to that of the prominent color, thus
        binning together similar colors. In this way it goes down the list
        until all colors present have been binned.
        The function returns a new ColorList object, as well as a dictionary
        that tells the user which colors have been binned together. This
        dictionary is of the form {major_color: [list, of, minor, colors]}.
        """
        colorlist = self.colorlist
        clustered_colorlist = []
        synonymous_colors = {}

        for color in colorlist:
            color_copy = color
            synonymous_colors[tuple(color[0:2])] = []

            # Store color as Lab-color
            color_rgb = sRGBColor(
                color[0], color[1], color[2], is_upscaled=True)
            color_lab = convert_color(color_rgb, LabColor)

            # Loop through all the colors that are less prominent than
            # the current color
            for color_compare in colorlist[colorlist.index(color)+1:]:

                # Store color as Lab-color
                color_compare_rgb = sRGBColor(
                    color_compare[0], color_compare[1], color_compare[2],
                    is_upscaled=True)
                color_compare_lab = convert_color(color_compare_rgb, LabColor)

                # Calculate the distance in color space
                delta = delta_e_cie2000(color_lab, color_compare_lab)

                # If distance is smaller than threshold, label as similar
                if (delta < radius):

                    # Add up pixel counts
                    color_copy[3] += color_compare[3]

                    # Remove color from the list we are looping over
                    colorlist.remove(color_compare)

                    synonymous_colors[tuple(color[0:2])].append(
                        color_compare[0:2])

            # Add color with updated pixel count to new list
            clustered_colorlist.append(color_copy)

        clustered_colorlist.sort(key=lambda tup: tup[3], reverse=True)

        BinnedColorList = ColorList.from_list(clustered_colorlist)
        return BinnedColorList, synonymous_colors

    def colors(self):
        """
        Returns a numpy array similar to a ColorList, but without pixel counts.
        """
        colorlist_copy = self.colorlist
        for color in colorlist_copy:
            del color[3]
        return colorlist_copy

    def to_dataframe(self):
        """
        Converts the ColorList to a DataFrame (with a single row.)
        """
        colordict = {'('+str(x[0])+','+str(x[1])+',' +
                     str(x[2])+')': x[3] for x in self.colorlist}
        df = pd.DataFrame(colordict, index=[0])
        return df

    def palette(self, uniquecolor, barwidth=500, barheight=100):
        """
        Generate a palette image with horizontal bands in the colors found in
        the ColorList. The RGB hex code is overlayed as well.
        """
        paletteimg = np.empty((0, 3))
        for color in self.colorlist:
            pixels = np.empty((barwidth*barheight, 3))
            pixels[...] = color[0:2]
            paletteimg = np.append(paletteimg, pixels)

        paletteimg = np.reshape(
            paletteimg, (paletteimg.size/(3*barwidth), barwidth, 3))

        counter = 0
        for color in self.colorlist:
            y = barheight/2 + barheight*counter
            cv2.putText(paletteimg,
                        str(rgb2hex([x/255.0 for x in color[0:2:-1]])),
                        (10, y),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1.5,
                        uniquecolor)
            counter += 1

        return paletteimg


class Map:
    """
    This class represents a map object, which is essentially an image,
    with the following additional properties:
     - An OpenCV histogram of colors found in the image
     - A ColorList representation of this histogram
    When Dynamic Color Binning is run on the Map, additional properties
    are created:
     - A binned version of the ColorList
     - A dictionary mapping major colors to a list of synonymous minor colors
    """

    def __init__(self, image):
        self.img = image
        self.histogram = cv2.calcHist([self.img], [0, 1, 2], None, [
                                      256, 256, 256], [0, 256, 0, 256, 0, 256])
        self.colorlist = ColorList(self.histogram)
        self.binned_colorlist = None
        self.synonymous_colors = None

    def run_dynamic_binning(self, radius=10):
        """
        This function simply runs DCB on the Map's ColorList
        """
        self.binned_colorlist, self.synonymous_colors = \
            self.colorlist.dynamic_binning(radius)

    def dataframe(self):
        """
        This function returns the DataFrame representation of
        the Map's ColorList
        """
        return self.colorlist.to_dataframe()

    def binned_dataframe(self):
        """
        Like dataframe, but returns the binned version
        """
        if (self.synonymous_colors is None):
            self.run_dynamic_binning()
        return self.binned_colorlist.to_dataframe()

    def clean(self):
        """
        This function takes the Map's image and replaces all minor
        colors with their major synonym.
        """
        if (self.synonymous_colors is None):
            self.run_dynamic_binning()
        cleaned_img = self.img.copy()
        for major_color in self.binned_colorlist.colors():
            for minor_color in self.synonymous_colors[major_color]:
                cleaned_img[np.where(
                    (cleaned_img == minor_color).all(axis=2))] = major_color
        return cleaned_img


class SetOfMaps:
    """
    This class represents a set of Map objects. It is essentially
    nothing but a list of such objects, but each identified with a
    given label. This label is necessary for the main functionality
    of the class, which is extract color data from the Map objects
    into a pandas DataFrame. This is the final form of the data that
    we are looking for.
    """

    def __init__(self, list_of_maps=[], list_of_labels=[]):
        self.maps = list_of_maps
        self.labels = list_of_labels
        self.mapping = dict(zip(self.labels, self.maps))
        if (len(self.maps) > 0):
            dfs = []
            # Loop through Maps. For each, get the binned dataframe and attach
            # the label. Then concatenate them all into one big dataframe.
            for label in self.labels:
                df = self.mapping[label].binned_dataframe().rename(index={
                    0: label})
                dfs.append(df)
            self.dataframe = pd.concat(dfs)

    def add_map(self, new_map, new_label):
        """
        This function adds a new Map, with the supplied label, to the
        SetOfMaps object. Note that if one wants to work with the dataframe,
        one should call self.update_dataframe() after adding new Maps.
        """
        self.maps.append(new_map)
        self.labels.append(new_label)
        self.mapping[new_label] = new_map

    def update_dataframe(self):
        """
        This function recreates the dataframe, in the same way that it is done
        in the constructor. This is useful if more Maps are added to the Set.
        """
        dfs = []
        for label in self.labels:
            df = self.mapping[label].binned_dataframe().rename(index={
                0: label})
            dfs.append(df)
        self.dataframe = pd.concat(dfs)

    def bin_dataframe(self, radius):
        """
        This function looks at the Set's dataframe and checks whether there are
        columns that are closer together than _radius_ in colorspace. Such
        columns are then merged.

        The algorithm is similar to the DCB algorithm itself, which is heavily
        commented in the ColorList class.
        """
        cols = list(self.dataframe)

        # Perform checking
        for col in cols:
            colbgr = literal_eval(col)
            color = sRGBColor(colbgr[0], colbgr[1],
                              colbgr[2], is_upscaled=True)
            color_lab = convert_color(color, LabColor)

            for compcol in cols[cols.index(col)+1:]:
                compcolbgr = literal_eval(compcol)
                compcolor = sRGBColor(
                    compcolbgr[0], compcolbgr[1], compcolbgr[2],
                    is_upscaled=True)
                compcolor_lab = convert_color(compcolor, LabColor)
                delta = delta_e_cie2000(color_lab, compcolor_lab)
                if (delta < radius):
                    self.dataframe[col].fillna(
                        self.dataframe[compcol], inplace=True)
                    del self.dataframe[compcol]
                    cols.remove(compcol)

        # Clean up dataframe (sorting columns, setting NaN to 0)
        # self.dataframe.sort_index(inplace=True)
        self.dataframe.fillna(0, inplace=True)
        self.dataframe = self.dataframe.reindex_axis(sorted(
            self.dataframe.columns,
            key=lambda x: self.dataframe[x].sum(), reverse=True), axis=1)

    def filter_dataframe(self, thresh):
        """
        This function removes any columns from the dataframe whose largest
        pixel count is smaller than thresh.
        """
        self.dataframe = self.dataframe.loc[:, self.dataframe.max() > thresh]

    def hexheader_dataframe(self):
        """
        This function returns a copy of the object's dataframe, but with the
        BGR headers replaced by hex color codes.
        """
        dataframe_copy = self.dataframe.copy()
        cols = list(self.dataframe)
        hexcols = [str(rgb2hex([y/255.0 for y in list(literal_eval(x))[::-1]]))
                   for x in cols]
        dataframe_copy.columns = hexcols
        return dataframe_copy

    def palette(self, uniquecolor, barwidth=500, barheight=100):
        """
        Generate a palette image with horizontal bands in the colors found in
        the set's dataframe. The RGB hex code is overlayed as well.
        """
        # Create empty image
        paletteimg = np.empty((0, 3))

        # Get colors from dataframe column headers
        cols = list(self.dataframe)
        for col in cols:
            color = list(literal_eval(col))

            # Create a new bar filled with just the color and append to palette
            pixels = np.empty((barwidth*barheight, 3))
            pixels[...] = color
            paletteimg = np.append(paletteimg, pixels)

        # Create a 2D image array from the flat list
        paletteimg = np.reshape(
            paletteimg, (paletteimg.size/(3*barwidth), barwidth, 3))

        # Overlay hex color codes as text
        counter = 0
        for col in cols:
            color = list(literal_eval(col))
            y = barheight/2 + barheight*counter
            cv2.putText(paletteimg,
                        str(rgb2hex([x/255.0 for x in color[::-1]])),
                        (10, y),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1.5,
                        uniquecolor)
            counter += 1

        return paletteimg
