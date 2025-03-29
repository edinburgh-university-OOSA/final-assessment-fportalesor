

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

  def writeTiff(self, data, x, y, res, filename="chm.tif",epsg=3031):
    '''
    Write a geotiff from a raster layer
    '''
    # Reproject the coordinates from EPSG:4326 (lat/lon) to EPSG:3031 (Antarctic Polar Stereographic)
    x_reprojected, y_reprojected = self.reprojectCoordinates(x, y, in_epsg=4326, out_epsg=epsg)
    
    # determine bounds
    minX=np.nanmin(x_reprojected) #nanmin and nanmax to avoid nans
    maxX=np.nanmax(x_reprojected)
    minY=np.nanmin(y_reprojected)
    maxY=np.nanmax(y_reprojected)

    # determine image size
    nX=int((maxX-minX)/res+1)
    nY=int((maxY-minY)/res+1)

    # pack in to array
    imageArr=np.full((nY,nX),-999.0)

    # calculate the raster pixel index in x and y
    xInds=np.array(np.floor((x_reprojected -np.min(x_reprojected))/res),dtype=int)   # need to force to int type
    yInds=np.array(np.floor((np.max(y_reprojected)-y_reprojected)/res),dtype=int)

    # this is a simple pack which will assign a single footprint to each pixel
    imageArr[yInds,xInds]=data

    # set geolocation information (note geotiffs count down from top edge in Y)
    geotransform = (minX, res, 0, maxY, 0, -1*res)

    # load data in to geotiff object
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nX, nY, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)    # specify coords

    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(epsg)                # WGS84 lat/long
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


#######################################################

