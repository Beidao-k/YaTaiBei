import pandas as pd
import os

# 定义文件路径
input_file = os.path.join("Pre_Data", "SIO2.csv")  # 输入文件路径
n_output_file = os.path.join("Pre_Data", "SIO2_n.csv")  # n数据输出路径
k_output_file = os.path.join("Pre_Data", "SIO2_k.csv")  # k数据输出路径

# 确保Pre_Data文件夹存在（如果不存在则创建）
os.makedirs("Pre_Data", exist_ok=True)

try:
    # 读取原始CSV文件（假设无表头，第一列是wl，第二列是n+k拼接数据）
    # 如果原始文件有表头（如"wl", "value"），请将header=0改为header=None，或根据实际表头调整
    df = pd.read_csv(input_file, header=None, names=["wl", "value"])  # 自定义列名

    # 关键步骤：区分n和k数据
    # 假设n和k数据的分界是"wl"的重复出现（因为k数据在n下方，wl会重新开始）
    # 找到第二个"wl"出现的位置（如果原始文件有表头，需调整逻辑）
    wl_column = df["wl"].astype(str)  # 转换为字符串避免数值比较问题
    split_index = None

    # 方法1：如果wl是数值型，且n和k的wl范围相同（重复出现），找到wl重复的起始索引
    # 找到第一个元素的位置，第二个相同元素的位置即为分界点
    first_wl = wl_column.iloc[0]
    for i in range(1, len(wl_column)):
        if wl_column.iloc[i] == first_wl:
            split_index = i
            break

    # 方法2：如果没有重复的wl（n和k的wl连续排列），请手动指定分界行数
    # 例如：split_index = 100  # 假设前100行是n数据，后面是k数据
    # 请根据你的实际数据行数修改上面的数值

    # 检查是否找到分界点
    if split_index is None:
        raise ValueError("未找到n和k数据的分界点！请检查数据格式，或手动设置split_index")

    # 分割n和k数据
    df_n = df.iloc[:split_index].copy()  # 前半部分是n数据
    df_k = df.iloc[split_index:].copy()  # 后半部分是k数据

    # 重命名列名（明确存储的是n还是k）
    df_n.rename(columns={"value": "n"}, inplace=True)
    df_k.rename(columns={"value": "k"}, inplace=True)

    # 重置k数据的索引（可选，使索引从0开始）
    df_k.reset_index(drop=True, inplace=True)

    # 保存到新的CSV文件
    df_n.to_csv(n_output_file, index=False, header=True)
    df_k.to_csv(k_output_file, index=False, header=True)

    print(f"数据分离完成！")
    print(f"n数据已保存到：{n_output_file}")
    print(f"k数据已保存到：{k_output_file}")
    print(f"n数据行数：{len(df_n)}")
    print(f"k数据行数：{len(df_k)}")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}")
    print("请检查文件路径是否正确，确保SIO2.csv在Pre_Data文件夹下")
except Exception as e:
    print(f"程序出错：{str(e)}")
    print("\n解决建议：")
    print("1. 检查SIO2.csv的格式是否为两列（wl和value）")
    print("2. 如果n和k的wl没有重复，请手动设置split_index（例如split_index=100）")
    print("3. 确保CSV文件没有乱码，分隔符为逗号")