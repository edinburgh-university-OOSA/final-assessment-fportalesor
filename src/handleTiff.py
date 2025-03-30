'''
A class to handle geotiffs
'''

#######################################################
# import necessary packages

from pyproj import Transformer # package for reprojecting data
from osgeo import gdal             # package for handling geotiff data
from osgeo import osr              # package for handling projection information
import numpy as np
from processLVIS import lvisGround

#######################################################

class tiffHandle(lvisGround):
  '''
  Class to handle geotiff files
  '''

  ########################################

  def reprojectCoordinates(self, x, y, in_epsg=4326, out_epsg=3031):
    '''
    Reproject the coordinates from one EPSG to another
    '''
    # Initialize the transformer for the given projections
    transformer = Transformer.from_crs(in_epsg, out_epsg, always_xy=True)

    # Perform the transformation (note the new API with .transform())
    x_new, y_new = transformer.transform(x, y)

    return x_new, y_new

  def writeTiff(self, data, x, y, res, filename="chm.tif", in_epsg=4326, out_epsg=3031, reproject=True):
    '''
      Writes a geotiff from raster data.

        Args:
            data (numpy.ndarray): Array of data values to write to raster
            x (array-like): X-coordinates of data points
            y (array-like): Y-coordinates of data points
            res (float): Resolution of output raster in target units
            filename (str, optional): Output filename. Defaults to "chm.tif".
            in_epsg (int, optional): EPSG code of input coordinates. Defaults to 4326.
            out_epsg (int, optional): EPSG code for output geotiff. Defaults to 3031.
            reproject (bool, optional): Whether to reproject coordinates. Defaults to True.

        Returns:
            None: Writes file to disk

        Note:
            When reproject=False, the output EPSG will match the input EPSG.
            NoData values are set to -999.
    '''
    if reproject:
      # Reproject the coordinates from input EPSG to target EPSG
      x_processed, y_processed = self.reprojectCoordinates(x, y, in_epsg=in_epsg, out_epsg=out_epsg)
    else:
      # Use coordinates as-is
      x_processed, y_processed = x, y
      out_epsg = in_epsg  # Use input EPSG as output EPSG when not reprojecting
    
    # determine bounds
    minX=np.min(x_processed)
    maxX=np.max(x_processed)
    minY=np.min(y_processed)
    maxY=np.max(y_processed)

    # determine image size
    nX=int((maxX-minX)/res+1)
    nY=int((maxY-minY)/res+1)

    # pack in to array
    imageArr=np.full((nY,nX),-999.0)

    # Loop through the LiDAR data points
    for i in range(len(x_processed)):
        x_ind = int(np.floor((x_processed[i] - minX) / res))
        y_ind = int(np.floor((maxY - y_processed[i]) / res))  # Invert y-axis for GeoTIFF convention

        # Ensure the indices are within the bounds of the raster
        if 0 <= x_ind < nX and 0 <= y_ind < nY:
            if imageArr[y_ind, x_ind] == -999.0:
                imageArr[y_ind, x_ind] = data[i]  # First data point for this pixel
            else:
                # Average the values (add the new value and divide by the number of points)
                imageArr[y_ind, x_ind] = (imageArr[y_ind, x_ind] + data[i]) / 2

    # set geolocation information (note geotiffs count down from top edge in Y)
    geotransform = (minX, res, 0, maxY, 0, -1*res)

    # load data in to geotiff object
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nX, nY, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)    # specify coords

    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(out_epsg)           # Use the output EPSG code
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file

    dst_ds.GetRasterBand(1).WriteArray(imageArr)  # write image to the raster
    dst_ds.GetRasterBand(1).SetNoDataValue(-999)  # set no data value
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

    print("Image written to",filename)


  ########################################

  def readTiff(self,filename):
    '''
    Read a geotiff in to RAM
    '''

    # open a dataset object
    ds=gdal.Open(filename)
    # could use gdal.Warp to reproject if wanted?

    # read data from geotiff object
    self.nX=ds.RasterXSize             # number of pixels in x direction
    self.nY=ds.RasterYSize             # number of pixels in y direction
    # geolocation tiepoint
    transform_ds = ds.GetGeoTransform()# extract geolocation information
    self.xOrigin=transform_ds[0]       # coordinate of x corner
    self.yOrigin=transform_ds[3]       # coordinate of y corner
    self.pixelWidth=transform_ds[1]    # resolution in x direction
    self.pixelHeight=transform_ds[5]   # resolution in y direction
    # read data. Returns as a 2D numpy array
    self.data=ds.GetRasterBand(1).ReadAsArray(0,0,self.nX,self.nY)