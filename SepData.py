import os
import pandas as pd
import glob


def process_csv_files(directory_path):
    """
    处理指定目录下的CSV文件，将wl,n和wl,k分离保存为CSV文件
    确保科学计数法的数字以普通格式显示
    """
    # 查找目录下所有的CSV文件
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"在目录 {directory_path} 中没有找到CSV文件")
        return

    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查列数
            if df.shape[1] != 2:
                print(f"文件 {file_path} 的列数不是2列，跳过处理")
                continue

            # 获取列名
            columns = df.columns.tolist()

            # 检查列名是否符合预期
            if len(columns) < 2:
                print(f"文件 {file_path} 的列名不足，跳过处理")
                continue

            # 第一列应该是wl，第二列是n/k
            wl_col = columns[0]
            nk_col = columns[1]

            # 分离n和k数据
            # 假设n在k上面，即n在前半部分，k在后半部分
            total_rows = len(df)
            n_rows = total_rows // 2
            k_rows = total_rows - n_rows

            # 提取wl,n数据
            wl_n_df = df.iloc[:n_rows].copy()
            # 重命名列名为 wl, n
            wl_n_df.columns = ['wl', 'n']

            # 提取wl,k数据 - 从n_rows开始到结束
            wl_k_df = df.iloc[n_rows:].copy()
            # 重置索引
            wl_k_df = wl_k_df.reset_index(drop=True)
            # 重命名列名为 wl, k
            wl_k_df.columns = ['wl', 'k']

            # 生成新的文件名
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]

            n_file_name = f"{name_without_ext}_n.csv"
            k_file_name = f"{name_without_ext}_k.csv"

            n_file_path = os.path.join(directory_path, n_file_name)
            k_file_path = os.path.join(directory_path, k_file_name)

            # 保存为CSV文件，禁用科学计数法
            wl_n_df.to_csv(n_file_path, index=False, float_format='%.10f')
            wl_k_df.to_csv(k_file_path, index=False, float_format='%.10f')

            print(f"处理完成: {file_name}")
            print(f"  - 生成: {n_file_name} (包含 {len(wl_n_df)} 行, 列名: wl, n)")
            print(f"  - 生成: {k_file_name} (包含 {len(wl_k_df)} 行, 列名: wl, k)")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")


def main():
    # 指定目录路径
    directory_path = r"D:\Python_Code\YaTaiBei\Pre_Data"

    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return

    print(f"开始处理目录: {directory_path}")
    process_csv_files(directory_path)
    print("所有文件处理完成！")


if __name__ == "__main__":
    main()