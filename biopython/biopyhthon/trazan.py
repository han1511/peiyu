# convert_to_tarzan.py

def convert_to_tarzan(host_file, virus_file, mapping_file, output_file):
    # 读取宿主树
    with open(host_file, "r", encoding="utf-8") as f:
        host_tree = f.read().strip()

    # 读取病毒树
    with open(virus_file, "r", encoding="utf-8") as f:
        virus_tree = f.read().strip()

    # 读取宿主-病毒映射
    associations = []
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    host, parasite = parts[0], parts[1]
                    associations.append(f"    {host} : {parasite}")

    # 写入 TARZAN 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("<Tarzan>\n")
        f.write("  <hostTree>\n")
        f.write(f"    {host_tree}\n")
        f.write("  </hostTree>\n\n")
        f.write("  <parasiteTree>\n")
        f.write(f"    {virus_tree}\n")
        f.write("  </parasiteTree>\n\n")
        f.write("  <associations>\n")
        f.write("\n".join(associations))
        f.write("\n  </associations>\n")
        f.write("</Tarzan>\n")

    print(f"✅ TARZAN 文件已生成: {output_file}")


if __name__ == "__main__":
    # 这里改成你自己的文件名
    host_file = "H:\\研究生\\硕士研究生\\陈雨虹\\共进化缠结图的绘制\\host.nwk"
    virus_file = "H:\\研究生\\硕士研究生\\陈雨虹\\共进化缠结图的绘制\\virus.nwk"
    mapping_file = "H:\\研究生\\硕士研究生\\陈雨虹\\共进化缠结图的绘制\\mapping.txt"
    output_file = "H:\\研究生\\硕士研究生\\陈雨虹\\共进化缠结图的绘制\\output.tarzan"

    convert_to_tarzan(host_file, virus_file, mapping_file, output_file)
