import subprocess
import time
import time

import select

# 启动一个交互式命令行
process = subprocess.Popen(["cmd"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#process = subprocess.Popen(cmd_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

FeaNum = '20' + '\n'
# 定义要执行的多个命令
commands = [
    "cd Estimate\n",
    "Rscript .\RadioML_features.R -p TIANTAN -t 5 -n 1000 -R prefix_1000_5_training.RData\n",
    FeaNum
]

# 发送多个命令到命令行
for command in commands:
    process.stdin.write(command)
    process.stdin.flush()  # 刷新缓冲区以确保命令被发送
    # 读取命令的输出
    if process.stdout.readline():
        output_line = process.stdout.readline()
        if not output_line:  # 如果没有更多输出，则退出循环
            #process.stdin.write("exit\n")
            print("Finished")
            break
        print(output_line.strip())
    time.sleep(1)  # 等待一秒钟以确保命令有足够的时间执行
    if command == FeaNum:  # 如果没有更多输出，则退出循环
        print("Finished")

output_line = process.stdout.readline()
start_time = time.time()
while output_line:

    print(output_line.strip())

    # 设置超时时间为1秒
    #print(time.time() - start_time)
    if time.time() - start_time >= 0.85:
        break

    output_line = process.stdout.readline()

"""
# 读取命令的输出
while True:
    output_line = process.stdout.readline()
    if not output_line:  # 如果没有更多输出，则退出循环
        process.stdin.write("exit\n")
        print("Finished")
        break
    print(output_line.strip())
sys.exit(process)
"""


