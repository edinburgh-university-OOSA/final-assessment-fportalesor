from _bootstrap import *
import argparse
import os
from src.sectionProcessing import LVISSectionProcessor
from src.plotting import plotLVIS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LVIS data to generate DEM.")
    parser.add_argument('-f', '--filename', type=str, required=True,
                      help="Input LVIS data file path")
    parser.add_argument('-s', '--section_factor', type=int, required=True,
                      help="Number of sections along each axis (e.g., 10 for 10x10 grid)")
    parser.add_argument('-r', '--resolution', type=int, required=True,
                      help="Output resolution in meters")
    parser.add_argument('--keep-sections', action='store_true',
                      help="Keep temporary section files (disables automatic cleanup)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # PIG bbox in EPSG:4326 (-180 to 180)
    PIG_BBOX = (-102.5, -75.7, -95.0, -73.8)

    # Initialise processor with cleanup enabled by default
    processor = LVISSectionProcessor(
        filename=args.filename,
        section_factor=args.section_factor,
        resolution=args.resolution,
        clean_sections=not args.keep_sections,
        bbox=PIG_BBOX
    )
    
    # Process all sections and create mosaic
    processor.process_all_sections()

    # Generate the input TIFF path automatically
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    year = os.path.basename(os.path.dirname(args.filename))
    tiff_path = os.path.join(
        "processed_data",
        f"{base_name}_{year}",
        f"{base_name}_mosaic.tif"
    )

    # Load and plot the DEM
    dem = plotLVIS(args.filename)
    dem.plotDEM(tiff_path)