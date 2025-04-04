from _bootstrap import *
import argparse
import glob
import os
from src.sectionProcessing import LVISSectionProcessor
from src.handleTiff import tiffHandle
from src.plotting import plotLVIS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LVIS data to generate DEMs.")

    parser.add_argument('-y', '--year', type=str, required=True,
                      help="Year of data collection (e.g., 2009, 2015)")
    parser.add_argument('-s', '--section-factor', type=int, required=True,
                      help="Number of sections along each axis (e.g., 10 for 10x10 grid)")
    parser.add_argument('-r', '--resolution', type=int, required=True,
                      help="Output resolution in metres")
    parser.add_argument('--keep-sections', action='store_true',
                      help="Keep temporary section files (disables automatic cleanup)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # PIG bbox in EPSG:4326
    PIG_BBOX = (-102.5, -75.7, -95.0, -73.8)

    # Construct the full input folder path
    base_folder = '/geos/netdata/oosa/assignment/lvis/'
    input_folder = os.path.join(base_folder, str(args.year))

    # Get all HDF5 files (case-insensitive match)
    h5_files = glob.glob(os.path.join(input_folder, '*.h5')) + \
               glob.glob(os.path.join(input_folder, '*.H5'))

    if not h5_files:
        print(f"No .h5 files found in {input_folder}")
        exit(1)

    print(f"Found {len(h5_files)} LVIS files to process")

    for h5_file in h5_files:
        print(f"\nProcessing file: {os.path.basename(h5_file)}")
        try:
            processor = LVISSectionProcessor(
                filename=h5_file,
                section_factor=args.section_factor,
                resolution=args.resolution,
                clean_sections=not args.keep_sections,
                bbox=PIG_BBOX
            )
            processor.process_all_sections()
        except Exception as e:
            print(f"Error processing {os.path.basename(h5_file)}: {str(e)}")
            continue

    print("\nAll files processed")

    # Load tiff Handler
    filename = '/geos/netdata/oosa/assignment/lvis/2009/ILVIS1B_AQ2009_1020_R1408_058456.h5'
    tiffHandler = tiffHandle(filename)
    
    # Define precise bbox for Pine Island Glacier (in EPSG:3031)
    BBOX = (-1.620e6, -0.300e6, -1.570e6, -0.230e6)

    # Create combined mosaics
    print(f"\nProcessing year: {args.year}")
    success = tiffHandler.create_combined_mosaic(year=args.year, bbox=BBOX)
        
    if success:
        print(f"Successfully processed year {args.year}")
    
    else:
        print(f"Failed to process year {args.year}")

    # Plot combined mosaics
    tiff_path = os.path.join('processed_data', f'combined_mosaic_{args.year}.tif')
    
    plotter = plotLVIS(filename)
    plotter.plotDEM(tiff_path)