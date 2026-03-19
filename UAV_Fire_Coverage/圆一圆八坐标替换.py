import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os


def convert_coordinates(csv_path, shp_path, name):
    print(f"\n[{name}] 开始处理...")

    # 根据你上次的日志，源数据的投影坐标系是 WGS 84 / UTM zone 47N
    # 对应的 EPSG 代码是 32647
    original_crs = "EPSG:32647"

    # ==========================================
    # 1. 读取并转换 SHP 文件 (火点)
    # ==========================================
    if not os.path.exists(shp_path):
        print(f"  ❌ 找不到 SHP 文件: {shp_path}")
    else:
        gdf_shp = gpd.read_file(shp_path)

        # 判断是不是已经转换过了
        if gdf_shp.crs and gdf_shp.crs.to_epsg() == 4326:
            print("  ⚠️ SHP 文件已经是经纬度 (EPSG:4326) 格式，跳过 SHP 转换。")
        else:
            # 强制设置原坐标系并转换
            gdf_shp.set_crs(original_crs, allow_override=True, inplace=True)
            gdf_shp_4326 = gdf_shp.to_crs(epsg=4326)
            gdf_shp_4326.to_file(shp_path, encoding='utf-8')
            print("  ✅ SHP 文件已成功转换为经纬度并覆盖原文件！")

    # ==========================================
    # 2. 读取并转换 CSV 文件 (圆心)
    # ==========================================
    if not os.path.exists(csv_path):
        print(f"  ❌ 找不到 CSV 文件: {csv_path}")
    else:
        df_csv = pd.read_csv(csv_path)

        # 检查有没有我们需要转换的列名 center_x 和 center_y
        if 'center_x' not in df_csv.columns or 'center_y' not in df_csv.columns:
            print(f"  ❌ CSV 文件中没有找到 center_x 或 center_y 列！")
        else:
            # 用 center_x 和 center_y 生成点，x是经度方向(水平)，y是纬度方向(垂直)
            geometry = [Point(x, y) for x, y in zip(df_csv['center_x'], df_csv['center_y'])]

            # 告诉 GeoPandas 这些点使用的是 UTM 47N (EPSG:32647)
            gdf_csv = gpd.GeoDataFrame(df_csv, geometry=geometry, crs=original_crs)

            # 转换成经纬度 (WGS84)
            gdf_csv_4326 = gdf_csv.to_crs(epsg=4326)

            # 生成强化学习代码需要的 latitude 和 longitude 列
            df_csv['longitude'] = gdf_csv_4326.geometry.x
            df_csv['latitude'] = gdf_csv_4326.geometry.y

            # 覆盖保存 CSV
            df_csv.to_csv(csv_path, index=False, encoding='utf-8')
            print("  ✅ CSV 文件已成功转换为经纬度并覆盖原文件！")
            print(f"  新的圆心坐标: 纬度 {df_csv['latitude'].iloc[0]:.6f}, 经度 {df_csv['longitude'].iloc[0]:.6f}")


if __name__ == "__main__":
    # 定义基础路径
    base_dir = r"E:\lzd\python\贪心圆\111-copilot-process-fire-data-and-cluster\output"

    # 圆一文件路径
    circle1_csv = os.path.join(base_dir, "circle_1_center.csv")
    circle1_shp = os.path.join(base_dir, "circle_1_points.shp")

    # 圆八文件路径
    circle8_csv = os.path.join(base_dir, "circle_8_center.csv")
    circle8_shp = os.path.join(base_dir, "circle_8_points.shp")

    # 执行转换
    convert_coordinates(circle1_csv, circle1_shp, "圆��� (Circle 1)")
    convert_coordinates(circle8_csv, circle8_shp, "圆八 (Circle 8)")

    print("\n🎉 全部处理完成！现在请回去运行 PPO 训练代码吧！")