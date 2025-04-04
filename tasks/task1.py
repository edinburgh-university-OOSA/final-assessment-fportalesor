from _bootstrap import *
import argparse
from src.plotting import plotLVIS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LVIS data to generate DEM.")
    parser.add_argument('-f', '--filename', type=str, required=True,
                      help="Input LVIS data file path")
    parser.add_argument('-i', '--index', type=int,
                      help="Waveform index to plot")
    return parser.parse_args()
    
def get_valid_index(lvis, wave_index=None):
    """Get and validate waveform index from user"""
    max_index = len(lvis.waves) - 1
    if wave_index is not None:
        if 0 <= wave_index <= max_index:
            return wave_index
        print(f"Warning: Index {wave_index} is out of range (0-{max_index})")
    
    while True:
        try:
            print(f"\nFile contains waveforms with indices 0 to {max_index}")
            user_input = input("Enter waveform index to plot (or 'q' to quit): ")
            if user_input.lower() == 'q':
                return None
            wave_index = int(user_input)
            if 0 <= wave_index <= max_index:
                return wave_index
            print(f"Please enter a number between 0 and {max_index}")
        except ValueError:
            print("Please enter a valid number")
    
if __name__=="__main__":

  args = parse_arguments()

  # Load the LVIS data
  lvis = plotLVIS(args.filename)

  # Get total number of waveforms
  total_waves = lvis.nWaves

  # Get and validate index
  wave_index = get_valid_index(lvis, args.index)
  if wave_index is None:
      exit()

  lvis.setElevations()
  elevations, waveform = lvis.getOneWave(wave_index)
  
  lvis.plotWaves(waveform, wave_index)