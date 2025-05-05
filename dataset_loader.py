"""
step1. load tiff and shp data 
step2. transform the crs: tifff - 4326; shp - 26910
step3. cut shp areas in tiff and plt
step4. resize the input in to 224*224 patch

"""

import rasterio
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio.windows
from shapely.geometry import box
import os
import json
import cv2
from rasterio.features import rasterize
import numpy as np
import torch 
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class DataLoader():
    def __init__(self, tiff_path = "./dataset/NWM_INT_PAINT.tiff", shp_path = "./dataset/NWM_paint/NWM_paint/paint.shp"):
 
        self.tiff_path = tiff_path
        self.shp_path = shp_path

        self.read_data() # To enable direct use of the self.shp_data variable later without needing external calls.

    def read_data(self):
        self.tiff_data, self.tiff_profile, self.tiff_crs = self._load_tiff()
        self.shp_data, self.shp_crs = self._load_shp()
        
        if self.tiff_crs != self.shp_crs:
            self.shp_data = self._convert_shp_crs()
    
    def _load_tiff(self):
        with rasterio.open(self.tiff_path) as src: 
            tiff_data = src.read(1) # Read the first band of data. The provided data contains only one band, which is elevation data.
            tiff_profile = src.profile
            tiff_crs = src.crs
        return tiff_data, tiff_profile, tiff_crs

    def _load_shp(self):
        shp_data = gpd.read_file(self.shp_path) # GeoDataFrame,with geopandas's attribution
        shp_crs = shp_data.crs
        return shp_data, shp_crs

    def _convert_shp_crs(self):
        converted_shp = self.shp_data.to_crs(self.tiff_crs)
        return converted_shp
    
    def plot_data(self, window_size = 5000):
        with rasterio.open(self.tiff_path) as src:
            width, height = src.width, src.height
        
        # Crop a region from the center point for visualization.
            center_x, center_y = width//2, height//2
            x_min, x_max = center_x - window_size//2, center_x + window_size//2
            y_min, y_max = center_y - window_size//2, center_y + window_size//2

            window = rasterio.windows.Window(x_min, y_min, window_size, window_size)
            small_tiff = src.read(1, window = window)
            transform = src.window_transform(window)

        bbox = box(transform.c, transform.f, transform.c + window_size*transform.a, transform.f + window_size*transform.e)
        clipped_shp = self.shp_data[self.shp_data.intersects(bbox)]
        fig, ax = plt.subplots(figsize = (8,6))
        ax.imshow(small_tiff, cmap = "terrain", extent = (
            transform.c,  # Top-left x-coordinate (in real-world geographic coordinates)
            transform.c + window_size * transform.a,  # Bottom-right x-coordinate
            transform.f + window_size * transform.e,  # Bottom-right y-coordinate (note that transform.e is usually negative)
            transform.f  # Top-left y-coordinate
        ))
        clipped_shp.plot(ax=ax, color="red", linewidth=0.5)
        plt.title("region Tiff & Shp check")
        plt.show()

class SegmentationPreprocessor:
    def __init__(self, data_loader, patch_size = 512, save_dir = "out_put"):
        self.tiff_data = data_loader.tiff_data ## tiff_data itself is bound to self within the DataLoader class.
        self.tiff_profile = data_loader.tiff_profile
        self.transform = self.tiff_profile["transform"]   
        self.crs = self.tiff_profile["crs"]
        self.shp_data = data_loader.shp_data
        
        self.patch_size = patch_size
        self.save_dir = save_dir
        self.image_dir = os.path.join(save_dir, "images")
        self.mask_dir = os.path.join(save_dir, "masks")
        self.label_map_path = os.path.join(save_dir, "label_mapping.json")
        
        os.makedirs(self.image_dir, exist_ok = True)
        os.makedirs(self.mask_dir, exist_ok= True)

        self.label2id = self._build_label_map()

    def _build_label_map(self, freq=5):
        ## First, preprocess the type data. The processing logic is documented in the README.

        label_clean_map = {
            "arow": "arrow", "ar":"arrow",
            "biike": "bike", "bikew": "bike", "bikwe": "bike", "bikw": "bike", "bike lane": "bike",
            "bus only": "bus",
            "bmp": "bump", "bumpp": "bump",
            "cw": "crosswalk", "cw'": "crosswalk", "cross": "crosswalk",
            "ds": "double_solid",
            "dsb": "double_solid_broken",
            "hashy": "hash", "hahs": "hash",
            "pedesterian": "pedestrian",
            "do not stop": "do_not_stop", "do nots stop": "do_not_stop",
            "sb": "single_broken",
            "ss": "single_solid", "ssl": "single_solid", "ssy": "single_solid", "solid": "ssingle_solid",
            "sl": "stopline", "stop line": "stopline", 
            "p": "parking"
        }
        # missing: rod, hov

        raw_labels = self.shp_data["type"].dropna().astype(str).str.lower()
        clean_labels = raw_labels.apply(lambda x: label_clean_map.get(x,x)) #.get(x, x) looks up the value corresponding to x in a dictionary. If the key exists, it returns the associated value; if not, it returns x itself.
        
        # Count the frequency and return only the values with a frequency greater than the specified threshold.
        label_count = clean_labels.value_counts()
        freq_labels = label_count[label_count>=freq].index.tolist()

        clean_labels = clean_labels.apply(lambda x: x if x in freq_labels else None)
        self.shp_data["type"] = clean_labels
        
        unique_labels = sorted(set(label for label in clean_labels if label is not None))
        label2id = {label: idx+1 for idx, label in enumerate(unique_labels)} ## unique_label contains None values that need to be preprocessed. To satisfy the requirements of CrossEntropy, labels should start from 0.
        
        with open(self.label_map_path, "w") as f:
            json.dump(label2id, f) # dump: Write a Python object (e.g., a dictionary) to a file in JSON format.
        
        # 为了统计频数，增加的内容
        label2freq = {label: int(label_count[label]) for label in unique_labels}
        label_freq_path = os.path.join(self.save_dir, "label_frequency.json")
        with open(label_freq_path, "w") as f:
            json.dump(label2freq, f, indent=4)
            
        return label2id

    def _rasterize_mask(self, shapes, out_shape, transform):
        # print("Rasterizing shape count:", len(shapes))
        # if shapes:
        #     print("First shape geometry:", shapes[0][0])
        #     print("First shape label ID:", shapes[0][1])

        mask = rasterize(
            shapes, 
            out_shape=out_shape,
            transform=transform,
            fill=0, # 除了shape外的background都是0
            dtype="uint8",
            all_touched=True
        )

        # print("Mask unique values:", np.unique(mask))
        return mask
    
    def generate_patches(self):
        height, width = self.tiff_data.shape
        count = 0 # Used to generate patch IDs

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):

                window = rasterio.windows.Window(x,y,self.patch_size,self.patch_size) # Create a rectangular window of patch size
                patch = self.tiff_data[y:y+self.patch_size, x:x+self.patch_size] # Extract a patch from the TIFF image; this corresponds to the window area and will be used for model training

                # Skip the patch if its size is not 512x512. According to test results, only three images do not meet this size requirement.
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    # print(f"Skipping patch {count:04d} due to image size: {patch.shape}") # debug
                    continue

                # calculate geospatial bounds of this patch
                patch_transform = rasterio.windows.transform(window, self.transform) # Apply affine transformation to convert the window's pixel coordinates to geographic coordinates
                patch_bounds = rasterio.windows.bounds(window, self.transform) # Returns (x_min, y_min, x_max, y_max)
                patch_box = box(*patch_bounds) # Construct a Shapely rectangular polygon to check for intersection with the SHP geometries

                # Clip SHP to patch bounds
                clipped = self.shp_data[self.shp_data.intersects(patch_box)].copy() 
                if clipped.empty: # Skip the patch if no road markings from the SHP file are contained within this patch box
                    continue 
                # convert geometries to (geometry, class_id) tuples
                ## Here we process the 'clipped' GeoDataFrame to extract geometry and class_id
                shapes = [
                    (geom, self.label2id[row["type"]])
                    for _, row in clipped.iterrows() 
                    if row["type"] is not None and row["type"] in self.label2id
                    for geom in row.geometry.geoms # Extract all polygons from each MultiPolygon
                    if hasattr(row.geometry, "geoms") # Check if the geometry has 'geoms' attribute; this if comes after for in list comprehension
                ] if clipped.geometry.iloc[0].geom_type == "MultiPolygon" else [
                    (row.geometry, self.label2id[row["type"]]) 
                    for _, row in clipped.iterrows()
                    if row["type"] is not None and row["type"] in self.label2id 
                ] # First check if the clipped geometry is a MultiPolygon (e.g., combined arrows). If so, flatten it to individual polygons.

                mask = self._rasterize_mask(shapes, out_shape=(self.patch_size, self.patch_size), transform=patch_transform) # Convert vector shapes to a 2D pixel mask

                # Skip patches without y label
                if mask.max() == 0:
                    continue 

                # save images and masks
                img_name = f"patch_{count:04d}.png"
                mask_name = f"patch_{count:04d}_mask.png"

                cv2.imwrite(os.path.join(self.image_dir, img_name), patch)
                cv2.imwrite(os.path.join(self.mask_dir, mask_name), mask)
                count += 1
        print(f"Generated {count} image-mask pairs in '{self.save_dir}'")