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


## Test
# data_loader = DataLoader()
# data_loader.plot_data()