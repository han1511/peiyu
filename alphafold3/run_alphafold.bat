@echo off

rem 激活虚拟环境
call D:\env\alphafold3\Scripts\activate.bat

rem 进入AlphaFold 3目录
cd "C:\Users\韩培钰\OneDrive\Desktop\冠状病毒刺突蛋白和ACE2\alphafold3-main\alphafold3-main"

rem 运行预测脚本
echo 请选择预测模式：
echo 1. 单个蛋白质预测
 echo 2. 蛋白质复合体预测
echo 3. 退出

set /p choice=请输入选项：

if "%choice%"=="1" (
    echo 正在进行单个蛋白质预测...
    python run_alphafold.py --fasta_paths=spike_protein.fasta --output_dir=output --model_preset=multimer_v3
) else if "%choice%"=="2" (
    echo 正在进行蛋白质复合体预测...
    python run_alphafold.py --fasta_paths=spike_ace2_complex.fasta --output_dir=output_complex --model_preset=multimer_v3
) else if "%choice%"=="3" (
    echo 退出脚本...
    exit
) else (
    echo 无效选项，请重新运行脚本并选择正确的选项。
    pause
    exit
)

rem 保持窗口打开
pause
