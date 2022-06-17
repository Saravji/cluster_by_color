# this code is based and expanded on the works by Christian Hill.
# refer to https://scipython.com/blog/converting-a-spectrum-to-a-colour/
# for the base code and explanations.
# modifications / additions are my own.
# this code is made available under the stipulations of BSD 3-Clause License
# dependencies: numpy. tested / developed with version 1.18.5

import numpy as np


class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf1931 = np.loadtxt('calibrate/cie-cmf-1931.txt', usecols=(1,2,3))
    cmf1964 = np.loadtxt('calibrate/cie-cmf-1964.txt', usecols=(1,2,3))
    D65 = np.loadtxt('calibrate/IlluminantD65_5nm_380_780.csv', usecols=(1))
    weight = np.sum(np.multiply(cmf1964[:, 1], D65))

    
    def __init__(self, red, green, blue, white, cmf=1964):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red = self.__xyz_from_xy__(*red)
        self.green  = self.__xyz_from_xy__(*green)
        self.blue = self.__xyz_from_xy__(*blue)
        self.white = white
        if cmf == 1964: self.cmf = self.cmf1964
        if cmf == 1931: self.cmf = self.cmf1931
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    
    def __xyz_from_xy__(self, x, y):
        """Return the vector (x, y, 1-x-y)."""
        return np.array((x, y, 1-x-y))

    
    def xy_to_xyz(self, x, y):
        return ColourSystem.__xyz_from_xy__(None, x, y)
    
    
    def xy_to_rgb(self, x, y, out_fmt=None):
        xyz = self.xy_to_xyz(x, y)
        return self.xyz_to_rgb(xyz, out_fmt)
    
    
    def xy_to_spec(self, x, y):
        xyz = self.xy_to_xyz(x, y)
        return self.xyz_to_spec(xyz)
    

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb
    
    
    def xyz_to_spec(self, xyz):
        result = np.multiply(xyz, ColourSystem.weight)
        result = np.sum(np.multiply(result, ColourSystem.cmf1964), axis=1)
        return result

    
    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    
    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    
    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)
    
    
    def sepc_to_xy(self, spec):
        X, Y, Z = self.spec_to_xyz(spec)
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        return x, y
    
    
illuminant_D65 = ColourSystem.xy_to_xyz(None, 0.3127, 0.3291)

cs_hdtv = ColourSystem(red=(0.67, 0.33),
                       green=(0.21, 0.71),
                       blue=(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=(0.63, 0.34),
                        green=(0.31, 0.595),
                        blue=(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=(0.64, 0.33),
                       green=(0.30, 0.60),
                       blue=(0.15, 0.06),
                       white=illuminant_D65)