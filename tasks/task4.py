from _bootstrap import *
import argparse
import os
from src.handleTiff import tiffHandle
from src.plotting import plotLVIS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LVIS data to generate DEMs.")

    parser.add_argument('-y', '--year', type=str, required=True,
                      help="Year of data collection (e.g., 2009, 2015)")
    parser.add_argument('-md', '--max_distance', type=int, default=None,
                      help="Maximum search distance for gap filling (pixels)")
    parser.add_argument('-s', '--smoothing', type=int, default=None,
                      help="Smoothing iterations for gap filling")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load tiff Handler
    filename = '/geos/netdata/oosa/assignment/lvis/2009/ILVIS1B_AQ2009_1020_R1408_058456.h5'
    tiffHandler = tiffHandle(filename)
    
    # Define precise bbox for Pine Island Glacier (in EPSG:3031)
    BBOX = (-1.620e6, -0.300e6, -1.570e6, -0.230e6)

    # Create combined mosaics and apply gap-filler algorithm
    print(f"\nProcessing year: {args.year}")
    success = tiffHandler.create_combined_mosaic(year=args.year,
                                                create_filled=True,
                                                max_distance=args.max_distance,
                                                smoothing=args.smoothing,
                                                bbox=BBOX)
        
    if success:
        print(f"Successfully processed year {args.year}")
    
    else:
        print(f"Failed to process year {args.year}")

    # Plot filled mosaics
    tiff_path = os.path.join('processed_data', f'filled_mosaic_{args.year}.tif')

    plotter = plotLVIS(filename)
    plotter.plotDEM(tiff_path)