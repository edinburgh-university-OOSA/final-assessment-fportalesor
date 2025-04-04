import tracemalloc
import subprocess
import os
import numpy as np
from processLVIS import lvisGround
from handleTiff import tiffHandle

class SectionProcessor:
    """Base class for section processing functionality"""
    def __init__(self, filename, section_factor, resolution, clean_sections=True, bbox=None):
        self.filename = filename
        self.section_factor = section_factor
        self.resolution = resolution
        self.memory_stats = []
        self.section_files = []
        self.output_dir = None
        self.base_name = None
        self.clean_sections = clean_sections,
        self.bbox = bbox

    def calculate_section_dimensions(self, bounds):
        """Calculate dimensions for each section"""
        x_min, y_min, x_max, y_max = bounds
        section_width = (x_max - x_min) / self.section_factor
        section_height = (y_max - y_min) / self.section_factor
        return x_min, y_min, section_width, section_height

    @staticmethod
    def convert_longitude(lon):
        """Convert longitude from 0–360 range to -180 to 180 range."""
        return lon - 360 if lon > 180 else lon

    def section_intersects_bbox(self, section_bbox):
        """Check if section intersects with target bbox (handling 0-360 to -180 conversion)"""
        if self.bbox is None:
            return True
            
        # Unpack section bounds (in 0-360)
        s_minlon, s_minlat, s_maxlon, s_maxlat = section_bbox
        
        # Unpack target bbox (in -180 to 180)
        t_minlon, t_minlat, t_maxlon, t_maxlat = self.bbox
        
        # Convert section bounds to -180-180 for comparison
        s_minlon_180 = self.convert_longitude(s_minlon)
        s_maxlon_180 = self.convert_longitude(s_maxlon)
        
        # Check intersection in converted coordinates
        return not (s_maxlon_180 < t_minlon or 
                   s_minlon_180 > t_maxlon or
                   s_maxlat < t_minlat or 
                   s_minlat > t_maxlat)
    
    def create_output_directory(self):
        """Create output directory based on input filename, placing it in processed_data folder"""
        path_parts = self.filename.split('/')
        year = path_parts[-2] if len(path_parts) > 1 else "unknown_year"
        self.base_name = os.path.splitext(os.path.basename(self.filename))[0]
    
        # Create path with processed_data as parent directory
        self.output_dir = os.path.join("processed_data", f"{self.base_name}_{year}")
    
        # Create directory (including any necessary parent directories)
        os.makedirs(self.output_dir, exist_ok=True)
    
        return self.output_dir

    def get_full_bounds(self):
        """Get bounds of the full dataset"""
        return lvisGround(self.filename, onlyBounds=True)

    def _cleanup_section_files(self):
        """Internal method to safely remove section files"""
        deleted = 0
        for section_file in self.section_files:
            try:
                os.remove(section_file)
                deleted += 1
            except OSError as e:
                print(f"Warning: Could not delete {section_file} - {e}")
    
        print(f"Cleaned up {deleted}/{len(self.section_files)} section files")
        self.section_files = []  # Clear the list regardless of deletion success

    def save_memory_stats(self):
        """Save memory peak values to a text file"""
        if not self.memory_stats:
            return
            
        stats_file = os.path.join(self.output_dir, f"{self.base_name}_memory_peaks.txt")
        with open(stats_file, 'w') as f:
            for peak in self.memory_stats:
                f.write(f"{peak:.2f}\n")
        print(f"Memory peaks saved to {stats_file}")
        print("\nMemory Usage Statistics:")
        print(f"Minimum peak memory: {min(self.memory_stats):.2f} MB")
        print(f"Maximum peak memory: {max(self.memory_stats):.2f} MB")
        print(f"Median peak memory: {np.median(self.memory_stats):.2f} MB")

class LVISSectionProcessor(SectionProcessor):
    """Concrete implementation for LVIS data processing"""
    def process_section(self, x0, y0, x1, y1, section_num):
        """Process a single LVIS section"""
        section_output = os.path.join(self.output_dir, f"section_{section_num:04d}.tif")
        
        try:
            tracemalloc.start()
            lvis = lvisGround(self.filename, minX=x0, minY=y0, maxX=x1, maxY=y1)
            
            if not hasattr(lvis, 'lon') or len(lvis.lon) == 0:
                print(f"No data points in section {section_num}")
                return False

            lvis.setElevations()
            lvis.findStats()
            threshold = lvis.meanNoise + 5 * lvis.stdevNoise
            lvis.denoise(threshold)
            lvis.estimateGround()

            tiff_handler = tiffHandle(self.filename)
            tiff_handler.writeTiff(lvis.zG, lvis.lon, lvis.lat, self.resolution, 
                                   filename=section_output)
            
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 ** 2)
            tracemalloc.stop()
            
            print(f"Section {section_num:04d} processed. Peak memory: {peak_mb:.2f} MB")
            self.memory_stats.append(peak_mb)
            self.section_files.append(section_output)
            return True
            
        except Exception as e:
            tracemalloc.stop()
            print(f"Error processing section {section_num:04d}: {str(e)}")
            return False

    def process_all_sections(self, clean_sections=True):
        """Process all sections in the dataset"""
        self.create_output_directory()
        full_bounds = self.get_full_bounds()
        x_min, y_min, section_width, section_height = self.calculate_section_dimensions(full_bounds.bounds)
        
        successful = 0
        total_sections = self.section_factor ** 2
        processed_sections = 0
        
        for i in range(self.section_factor):
            for j in range(self.section_factor):
                section_num = i * self.section_factor + j
                x0 = x_min + i * section_width
                x1 = x0 + section_width
                y0 = y_min + j * section_height
                y1 = y0 + section_height

                # Skip sections that don't intersect bbox
                if not self.section_intersects_bbox((x0, y0, x1, y1)):
                    continue
                
                if self.process_section(x0, y0, x1, y1, section_num):
                    successful += 1
                
                processed_sections +=1 
                progress = (processed_sections) / total_sections * 100
                print(f"Progress: {progress:.1f}% ({processed_sections}/{total_sections})")
        
        if successful > 0:
            final_output = os.path.join(self.output_dir, f"{self.base_name}_mosaic.tif")
            tiff_handler = tiffHandle(self.filename)
            tiff_handler.create_mosaic(self.section_files, final_output)

            if clean_sections:
                self._cleanup_section_files()
        
        self.save_memory_stats()
        
        print(f"\nProcessing complete. Successfully processed {successful}/{total_sections} sections")
        return successful

