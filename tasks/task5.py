from _bootstrap import *
import argparse
import os
from src.handleTiff import tiffHandle
from src.plotting import plotLVIS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate ice volume change between two DEMs.")
    
    parser.add_argument('-d1', '--dem1', type=str, required=True,
                      help="Path to first DEM (e.g., 2009 DEM)")
    parser.add_argument('-d2', '--dem2', type=str, required=True,
                      help="Path to second DEM (e.g., 2015 DEM)")
    parser.add_argument('-o', '--output', type=str, default="elevation_change.tif",
                      help="Output path for difference map (default: elevation_change.tif)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load tiff Handler
    filename = '/geos/netdata/oosa/assignment/lvis/2009/ILVIS1B_AQ2009_1020_R1408_058456.h5'
    tiffHandler = tiffHandle(filename)
    
    try:

        # 1. Create difference map
        # Set output directory and filename
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
        # Determine output path
        output_filename = args.output
        output_path = os.path.join(output_dir, output_filename)

        # 2. Get volume change
        volume_change, stats = tiffHandler.create_difference_dem(args.dem1, args.dem2, output_path)

        print(f"\nDifference map saved to {output_path}")
        
        print("\n=== Ice Volume Change Analysis ===")
        print(f"Total volume change: {volume_change/1e9:.4f} km³")
        print(f"Area analysed: {stats['area']/1e6:.2f} km²")
        print(f"Mean elevation change: {stats['mean']:.2f} m")
        print(f"Max thickening: {stats['max']:.2f} m")
        print(f"Max thinning: {stats['min']:.2f} m")
        
        # 3. Plot the difference
        plotter = plotLVIS(filename)
        plotter.plotDEM(output_path)
                        
        
    except Exception as e:
        print(f"Error in volume change analysis: {str(e)}")