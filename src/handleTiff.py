'''
A class to handle geotiffs
'''

#######################################################
# import necessary packages

from pyproj import Transformer # package for reprojecting data
from osgeo import gdal             # package for handling geotiff data
from osgeo import osr              # package for handling projection information
import numpy as np
import glob
import os
import subprocess
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
    self.ds = ds

    return self

  def create_mosaic(self, section_files, final_output):
    '''
    Creates a GeoTIFF mosaic from multiple input files using GDAL.
    Uses memory-efficient VRT processing and cleans up temporary files.

    Args:
        section_files: List of input GeoTIFF file paths (list[str])
        final_output: Output file path for mosaic (str)

    Returns:
        bool: True if successful, False if failed
    '''
    try:
        if not section_files:
            print("No section files found to mosaic")
            return False
            
        print(f"\nCreating mosaic from {len(section_files)} files...")
        
        # Create temporary VRT file
        vrt_file = "temp_mosaic.vrt"
        
        # Build VRT first
        cmd_vrt = [
            'gdalbuildvrt',
            '-hidenodata',
            '-vrtnodata', '-999',
            '-resolution', 'highest',
            '-r', 'nearest',
            vrt_file
        ] + section_files
        
        # Convert VRT to final TIFF
        cmd_translate = [
            'gdal_translate',
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',
            '-co', 'BIGTIFF=YES',
            '-co', 'TILED=YES',
            '-co', 'NUM_THREADS=ALL_CPUS',
            '-a_nodata', '-999',
            '--config', 'GDAL_CACHEMAX', '512',
            '--config', 'GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE',
            '--config', 'GDAL_NUM_THREADS', 'ALL_CPUS',
            vrt_file,
            final_output
        ]
        
        # Run both commands with error checking
        subprocess.run(cmd_vrt, check=True)
        subprocess.run(cmd_translate, check=True)
        
        # Clean up temporary VRT
        try:
            os.remove(vrt_file)
        except OSError:
            pass
        
        print(f"Successfully created mosaic: {final_output}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating mosaic: {e}")
        try:
            os.remove(final_output)
        except OSError:
            pass
        return False
    except Exception as e:
        print(f"Unexpected error creating mosaic: {e}")
        return False

  def fill_gaps(self, input_path, output_path, max_distance=45, smoothing=3):
    '''
    Fill gaps using GDAL's FillNodata algorithm
    
    Args:
        input_path (str): Input TIFF file path
        output_path (str): Output TIFF file path
        max_distance (int): Maximum search distance (pixels)
        smoothing (int): Smoothing iterations (0 for none)
        
    Returns:
        bool: True if successful, False if failed
    '''
    try:
        # 1. Open input file
        src_ds = gdal.Open(input_path)
        if src_ds is None:
            raise ValueError(f"Could not open {input_path}")

        # 2. Create output file
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.CreateCopy(
            output_path, 
            src_ds, 
            0,  # No special creation flags
            ['COMPRESS=LZW', 'BIGTIFF=YES', 'TILED=YES']
        )

        # 3. Fill gaps in each band
        for band_num in range(1, src_ds.RasterCount + 1):
            band = dst_ds.GetRasterBand(band_num)
            gdal.FillNodata(
                targetBand=band,
                maskBand=None,  # Optional mask band
                maxSearchDist=max_distance,
                smoothingIterations=smoothing
            )
            band.FlushCache()  # Ensure writes are saved

        # 4. Cleanup
        dst_ds = None  # Close file (required to save changes)
        src_ds = None
        return True

    except Exception as e:
        print(f"Gap filling failed: {str(e)}")
        # Cleanup partially created files
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

  def crop_tiff(self, input_path, output_path, bbox):
    '''
    Crop a TIFF file to the specified bounding box
    
    Args:
        input_path (str): Path to input TIFF file
        output_path (str): Path for output cropped TIFF
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax)
                      in the same CRS as the input TIFF
                      
    Returns:
        bool: True if successful, False if failed
    '''
    try:
        # Open input file
        ds = gdal.Open(input_path)
        if ds is None:
            raise ValueError(f"Could not open {input_path}")
            
        # Get geotransform and projection
        gt = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # Convert bbox to pixel coordinates
        xmin, ymin, xmax, ymax = bbox
        xoff = int((xmin - gt[0]) / gt[1])
        yoff = int((ymax - gt[3]) / gt[5])
        xsize = int((xmax - xmin) / gt[1])
        ysize = int((ymin - ymax) / gt[5])
        
        # Validate bounds
        xoff = max(0, xoff)
        yoff = max(0, yoff)
        xsize = min(ds.RasterXSize - xoff, xsize)
        ysize = min(ds.RasterYSize - yoff, ysize)
        
        # Read the subset
        data = ds.GetRasterBand(1).ReadAsArray(xoff, yoff, xsize, ysize)
        
        # Create output
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            xsize,
            ysize,
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        # Update geotransform
        new_gt = (
            gt[0] + xoff * gt[1],
            gt[1],
            gt[2],
            gt[3] + yoff * gt[5],
            gt[4],
            gt[5]
        )
        
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(projection)
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.GetRasterBand(1).SetNoDataValue(-999)
        out_ds.FlushCache()
        out_ds = None
        
        print(f"Cropped TIFF saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error cropping TIFF: {str(e)}")
        return False
    
  def create_combined_mosaic(self, year=2015, max_distance=None, 
                             smoothing=None, create_filled=False, bbox=None):
    '''
    Create a combined mosaic from all folders matching the year pattern
    
    Args:
        year (int): Year to search for in folder names
        max_distance (int): Maximum search distance for gap filling (pixels)
        smoothing (int): Smoothing iterations for gap filling
        create_filled (bool): Whether to create gap-filled version
        
    Returns:
        bool: True if successful, False if failed
    '''
    # Step 1: Find all folders ending with "_year"
    folders = glob.glob(f"processed_data/*_{year}")
    
    if not folders:
        print(f"No folders ending with '_{year}' found.")
        return False
    
    # Step 2: Collect all tif files
    tif_files = []
    
    for folder in folders:
        # Look for files ending with "_mosaic.tif" in each folder
        mosaic_files = glob.glob(os.path.join(folder, "*_mosaic.tif"))
        tif_files.extend(mosaic_files)
    
    if not tif_files:
        print("No '_mosaic.tif' files found in the folders.")
        return False
    
    # Create output paths within processed_data
    output_filename = os.path.join("processed_data", f"combined_mosaic_{year}.tif")
    filled_filename = os.path.join("processed_data", f"filled_mosaic_{year}.tif")
    
    # Step 3: Use create_mosaic function to combine them
    success = self.create_mosaic(tif_files, output_filename)
    
    if not success:
        return False
    
    # Determine which file will be processed (combined or filled)
    processing_filename = output_filename
    
    # Step 4: Apply gap filling if requested
    if create_filled:
        # Apply defaults if parameters not specified
        actual_max_dist = max_distance if max_distance is not None else 25
        actual_smoothing = smoothing if smoothing is not None else 3
        
        print(f"\nFilling gaps (distance: {actual_max_dist}, smoothing: {actual_smoothing})...")
        if not self.fill_gaps(output_filename, filled_filename, 
                            actual_max_dist, actual_smoothing):
            return False
        processing_filename = filled_filename
    
    # Step 5: Apply cropping if requested
    if bbox is not None:
        print("\nCropping mosaic to specified bounding box...")
        temp_filename = os.path.join("processed_data", "temp_cropped.tif")
        if not self.crop_tiff(processing_filename, temp_filename, bbox):
            return False
        
        # Replace original with cropped version
        os.replace(temp_filename, processing_filename)
        print(f"Cropping applied to: {processing_filename}")
    
    print(f"Successfully created final product: {processing_filename}")
    return True
  
    
  def create_difference_dem(self, dem1_path, dem2_path, output_path, resample_method='bilinear'):
    '''
    Create a difference DEM from two input DEMs in their overlapping area
    
    Args:
        dem1_path: Path to first DEM
        dem2_path: Path to second DEM
        output_path: Path for output difference DEM
        resample_method: Resampling method
    '''
    # Open both datasets
    self.readTiff(dem1_path)
    dem1_info = {
        'transform': (self.xOrigin, self.pixelWidth, 0, self.yOrigin, 0, self.pixelHeight),
        'projection': self.ds.GetProjection(),
        'bounds': (
            self.xOrigin,
            self.yOrigin + self.pixelHeight * self.nY,
            self.xOrigin + self.pixelWidth * self.nX,
            self.yOrigin
        ),
        'size': (self.nX, self.nY)
    }
    
    self.readTiff(dem2_path)
    dem2_info = {
        'transform': (self.xOrigin, self.pixelWidth, 0, self.yOrigin, 0, self.pixelHeight),
        'projection': self.ds.GetProjection(),
        'bounds': (
            self.xOrigin,
            self.yOrigin + self.pixelHeight * self.nY,
            self.xOrigin + self.pixelWidth * self.nX,
            self.yOrigin
        ),
        'size': (self.nX, self.nY)
    }

    # Check if projections match
    if dem1_info['projection'] != dem2_info['projection']:
        raise ValueError("DEMs must have the same projection")

    # Calculate overlapping area
    overlap_bounds = (
        max(dem1_info['bounds'][0], dem2_info['bounds'][0]),  # minx
        max(dem1_info['bounds'][1], dem2_info['bounds'][1]),  # miny
        min(dem1_info['bounds'][2], dem2_info['bounds'][2]),  # maxx
        min(dem1_info['bounds'][3], dem2_info['bounds'][3])   # maxy
    )

    # Check for overlap
    if overlap_bounds[0] >= overlap_bounds[2] or overlap_bounds[1] >= overlap_bounds[3]:
        raise ValueError("DEMs do not have any overlapping area")

    # Determine output resolution (use the finer resolution)
    res_x = min(abs(dem1_info['transform'][1]), abs(dem2_info['transform'][1]))
    res_y = min(abs(dem1_info['transform'][5]), abs(dem2_info['transform'][5]))

    # Warp both DEMs to the same grid
    warp_options = {
        'format': 'MEM',
        'outputBounds': overlap_bounds,
        'xRes': res_x,
        'yRes': res_y,
        'targetAlignedPixels': True,
        'resampleAlg': resample_method,
        'dstNodata': -999
    }

    # Align both DEMs
    dem1_aligned = gdal.Warp('', dem1_path, **warp_options)
    dem2_aligned = gdal.Warp('', dem2_path, **warp_options)

    # Get aligned arrays
    arr1 = dem1_aligned.GetRasterBand(1).ReadAsArray()
    arr2 = dem2_aligned.GetRasterBand(1).ReadAsArray()

    # Calculate differences on valid pixels
    valid_mask = (arr1 != -999) & (arr2 != -999)
    diff = np.full_like(arr1, np.nan)  # Use NaN for calculations
    diff[valid_mask] = arr2[valid_mask] - arr1[valid_mask]

    # Calculate statistics
    pixel_area = res_x * res_y
    volume_change = np.nansum(diff) * pixel_area

    stats = {
        'min': np.nanmin(diff),
        'max': np.nanmax(diff),
        'mean': np.nanmean(diff),
        'std': np.nanstd(diff),
        'area': np.sum(valid_mask) * pixel_area,
        'pixel_count': np.sum(valid_mask),
        'overlap_bounds': overlap_bounds,
        'resolution': (res_x, res_y),
        'aligned_dem1': dem1_aligned,
        'aligned_dem2': dem2_aligned,
        'difference_array': diff
    }
    
    # Calculate difference
    diff = np.full_like(arr1, -999, dtype=np.float32)
    valid_mask = (arr1 != -999) & (arr2 != -999)
    diff[valid_mask] = arr2[valid_mask] - arr1[valid_mask]

    # Prepare data for writeTiff
    x = np.linspace(
        stats['overlap_bounds'][0] + stats['resolution'][0]/2,
        stats['overlap_bounds'][2] - stats['resolution'][0]/2,
        diff.shape[1]
    )
    y = np.linspace(
        stats['overlap_bounds'][3] - stats['resolution'][1]/2,
        stats['overlap_bounds'][1] + stats['resolution'][1]/2,
        diff.shape[0]
    )
    
    # Get projection from one of the aligned DEMs
    srs = osr.SpatialReference()
    srs.ImportFromWkt(stats['aligned_dem1'].GetProjection())
    epsg = int(srs.GetAttrValue('AUTHORITY', 1))

    self.writeTiff(
        diff.flatten(),
        np.tile(x, diff.shape[0]),
        np.repeat(y, diff.shape[1]),
        stats['resolution'][0],
        filename=output_path,
        in_epsg=epsg,
        out_epsg=epsg,
        reproject=False
    )
    
    print(f"Created difference DEM in overlapping area: {output_path}")

    return volume_change, stats