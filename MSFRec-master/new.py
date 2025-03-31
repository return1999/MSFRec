def generate_inter_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # 写入表头
        outfile.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")

        # 初始化时间戳


        # 逐行读取输入文件
        for line in infile:
            # 去除行尾的换行符并按空格分割
            parts = line.strip().split()

            user_id = parts[0]
            timestamp = int(user_id)
            item_ids = parts[1:]

            # 对每个项目id进行处理
            for item_id in item_ids:
                # 写入转换后的数据到输出文件
                outfile.write(f"{user_id}\t{item_id}\t0\t{timestamp}\n")
                # 时间戳递增
                timestamp = timestamp + 1



# 调用函数生成新的data.inter文件
generate_inter_file('./data/Yelp.txt', './data/Yelp.inter')