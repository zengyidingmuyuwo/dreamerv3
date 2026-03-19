import geopandas as gpd

shp_path = r"E:\lzd\python\贪心圆\111-copilot-process-fire-data-and-cluster\output\circle_1_points.shp"
gdf = gpd.read_file(shp_path)

print("\n🔍 火点数据检查结果：")
print("一共读取到", len(gdf), "个火点。")
print("前三个火点的实际坐标是：")
for i in range(3):
    print(f"火点 {i+1}:", gdf.geometry.iloc[i])