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

class DataLoader():
    def __init__(self, tiff_path = "./dataset/NWM_INT_PAINT.tiff", shp_path = "./dataset/NWM_paint/NWM_paint/paint.shp"):
 
        self.tiff_path = tiff_path
        self.shp_path = shp_path

        self.read_data() # 为了后面能直接用self.shp_data变量，而不用外部调用

    def read_data(self):
        self.tiff_data, self.tiff_profile, self.tiff_crs = self._load_tiff()
        self.shp_data, self.shp_crs = self._load_shp()
        
        if self.tiff_crs != self.shp_crs:
            self.shp_data = self._convert_shp_crs()
    
    def _load_tiff(self):
        with rasterio.open(self.tiff_path) as src: 
            tiff_data = src.read(1) # 读第一波段的数据, 提供的数据只有一个波段，就是高程数据
            tiff_profile = src.profile
            tiff_crs = src.crs
        return tiff_data, tiff_profile, tiff_crs

    def _load_shp(self):
        shp_data = gpd.read_file(self.shp_path)
        shp_crs = shp_data.crs
        return shp_data, shp_crs

    def _convert_shp_crs(self):
        converted_shp = self.shp_data.to_crs(self.tiff_crs)
        return converted_shp
    
    def plot_data(self, window_size = 5000):
        with rasterio.open(self.tiff_path) as src:
            width, height = src.width, src.height
        
        # 从中心点剪裁一个区域画图
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
            transform.c,  # 左上角 x 坐标（实际地理坐标）
            transform.c + window_size * transform.a,  # 右下角 x 坐标
            transform.f + window_size * transform.e,  # 右下角 y 坐标（注意 transform.e 通常是负的）
            transform.f  # 左上角 y 坐标
        ))
        clipped_shp.plot(ax=ax, color="red", linewidth=0.5)
        plt.title("region Tiff & Shp check")
        plt.show()

class SegmentationPreprocessor:
    def __init__(self, data_loader, patch_size = 512, save_dir = "out_put"):
        self.tiff_data = data_loader.tiff_data ## tiff_data本身绑定在DataLoader这个类的self上
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
        os.makedirs(self.label_map_path, exist_ok= True)

        self.label2id = self._build_label_map()

    def _build_label_map(self):
        ### 这里对shp的type还要有一个预处理没有写进来
        unique_labels = self.shp_data["type"].unique()
        label2id = {label: idx+1 for idx, label in enumerate(sorted(unique_labels))}
        with open(self.label_map_path, "w") as f:
            json.dump(label2id, f) # dump：把python对象(e.g. 字典)以json的格式写入文件中
        return label2id

    def _rasterize_mask(self, shapes, out_shapes):
        return rasterize(
            shapes, 
            out_shape=out_shapes,
            transform=self.transform,
            fill=0, # 除了shape外的background都是0
            dtype="unit8",
        )
    
    def generate_patches(self):
        height, width = self.tiff_data.shape
        count = 0 # 用于生成patch编号

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                window = rasterio.windows.Window(x,y,self.patch_size,self.patch_size) # 创建了一个patch size的矩形窗口
                patch = self.tiff_data[y:y+self.patch_size, x:x+self.patch_size] # 从tiff中汲截取一个patch, 和window的范围对应，这个数据会输入模型进行训练

                # calculate geospatial bounds of this patch
                patch_transform = rasterio.windows.transform(window, self.transform) # 这里用到了上面定义的矩形窗口，并对这个区域的范围进行了affine变换(像素->地理坐标)
                patch_bounds = rasterio.windows.bounds(window, self.transform) # 返回的是(x_min, y_min, x_max, y_max)
                patch_box = box(*patch_bounds) #用来构造一个shapely的矩形polygon区域，用来和shp的集合做交集判断，判断就在下面

                # Clip SHP to patch bounds
                clipped = self.shp_data[self.shp_data.intersects(patch_box)].copy() 
                if clipped.empty(): # 如果这个patch_box中不包含shp文件中标记的road_marking的话，就略过它
                    continue 

                # convert geometries to (geometry, class_id) tuples
                shapes = [
                    (geom, self.label2id[row["type"]])
                    for _, row in clipped.iterrows()
                    for geom in row.geometry.geoms if hasattr(row.geometry, "geom")    
                ] if clipped.geometry.iloc[0].geom_type == "MultiPolygon" else [
                    (row.geometry, self.label2id[row["type"]]) for _, row in clipped.iterrows()
                ]

                mask = self._rasterize_mask(shapes, out_shapes=(self.patch_size, self.patch_size))

                # Skip patches without any label
                if mask.max() == 0:
                    continue 

                # save images and masks
                img_name = f"patch_{count:04d}.png"
                mask_name = f"patch_{count:04d}_mask.png"

                cv2.imwrite(os.path.join(self.image_dir, img_name), patch)
                cv2.imwrite(os.path.join(self.mask_dir, mask_name), mask)
                count += 1
        print(f"Generated {count} image-mask pairs in '{self.save_dir}'")
        ## 到目前为止对用于segmentation的数据进行了预处理，但是没有对type进行过滤，明天要开始看一下segmentation的模型了，并且把这个模块的注释补全

## Test
# data_loader = DataLoader()
# data_loader.plot_data()