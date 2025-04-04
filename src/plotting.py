import os
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from lvisClass import lvisData
from handleTiff import tiffHandle

class plotLVIS(lvisData):
    '''A class inheriting from lvisData with plotting capabilities'''
    
    def __init__(self, filename=None):
        super().__init__(filename)
        self.filename = filename 
        # Create plots directory if it doesn't exist
        self.plots_dir = os.path.join(os.path.dirname(os.getcwd()), 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plotWaves(self, wave, wave_index):
        '''Plot a specific waveform'''
        plt.plot(wave, label=f"Waveform {wave_index}")
        plt.xlabel("Points")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform (index={wave_index})", fontsize=10)
        
        output_png = os.path.join(self.plots_dir, f'waveform_{wave_index}.png')
        plt.savefig(output_png, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()
    
    def plotDEM(self, tiff_path, output_name=None):
        '''
        Plot a DEM stored as TIFF with basemap overlay
    
        Args:
            tiff_path: Path to input TIFF file
            output_name: Optional output filename (without extension)
        '''
        try:
            tiff_handler = tiffHandle(self.filename)
            tiff_handler.readTiff(tiff_path)
        
            # Get data and bounds
            layer = tiff_handler.data
            left = tiff_handler.xOrigin
            top = tiff_handler.yOrigin
            right = left + tiff_handler.nX * tiff_handler.pixelWidth
            bottom = top + tiff_handler.nY * tiff_handler.pixelHeight
        
            # Replace invalid values with NaN
            layer = np.where(layer == -999, np.nan, layer)
        
            # Create plot
            fig, ax = plt.subplots(figsize=(7, 7))
        
            # Plot DEM data (assuming EPSG:3031)
            img = ax.imshow(layer, 
                            cmap="RdYlBu_r",
                            extent=[left, right, bottom, top],
                            origin='upper',
                            alpha=0.7,
                            zorder=2
                            )

            # Add basemap
            ctx.add_basemap(ax, 
                            source=ctx.providers.Esri.WorldImagery,
                            crs="EPSG:3031",
                            attribution_size=6,
                            attribution="Tiles (C) Esri")

            # Format axes to use scientific notation
            ax.ticklabel_format(style='sci', axis='both', scilimits=(6,6))
            ax.xaxis.get_offset_text().set_fontsize(10)
            ax.yaxis.get_offset_text().set_fontsize(10)

            # Add colorbar
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label("Elevation (metres)")
        
            # Labels and title
            base_title = os.path.splitext(os.path.basename(tiff_path))[0]
            ax.set_title(f"DEM: {base_title}")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        
            # Save to plots directory
            if output_name is None:
                output_name = f"DEM_{base_title}"
            output_png = os.path.join(self.plots_dir, f"{output_name}.png")
            plt.savefig(output_png, dpi=300, bbox_inches="tight")
        
            plt.show()
            plt.close()
        
        except Exception as e:
            print(f"Error plotting DEM: {str(e)}")
            raise