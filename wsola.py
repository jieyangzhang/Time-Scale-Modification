# -*- coding: utf-8 -*-
# @Time    : 2021-04-30
# @Author  : Zhangjieyang
# @Email   : jieyangzhang@outlook.com
# @File    : wsola.py
# @Software: vscode
 
# 数学计算库
import numpy as np
# 读写音频会用到的库
from scipy.io import wavfile
import matplotlib.pyplot as plt
 
 
if __name__ == "__main__":
 
    # 读取音频文件
    freq, sig = wavfile.read("./audio.wav")
    len_data = len(sig)
 
    # -----------可配置项-start----------
    alpha = 2.18 # 时长调整比例 alpha = H_s / H_a
    # -----------可配置项--end-----------
 
    # ----------默认配置项-start---------
    rate = freq
    N = int(50 / 1000 * rate)  # 50ms时长的音频帧长
    H_s = int(N / 2)           # 输出帧间隔
    H_a = int(H_s / alpha)     # 原始信号分帧间隔
    # ----------默认配置项--end----------
    win = np.hanning(N) # 窗函数
    num_frames = int((len_data - N + H_a) / H_a) # 分帧数量
    print(H_a, H_s, num_frames, N)
 
    # ------初始化-start---------
    k = 0                                   # 分析帧的起始索引
    shift = 0                               # 最优分析帧检索偏移
    shift_max = int(N / 2)                       # 检索的最大范围
    output = np.zeros([num_frames * H_s])   # 输出信号
    shift_list = []
    score_list = []
    # ------初始化--end----------
 
    # 循环体
    for i in tqdm(range(num_frames)):
        # 分帧
        y_buffer = win * sig[k + shift: k + shift + N]
        x_tilde =  sig[k + shift + H_s: k + shift + H_s + N]
        # 检索区域
        retrieve_region = sig[k + H_a - shift_max: k + H_a + N + shift_max] if k + H_a - shift_max >= 0 else sig[k + H_a: k + H_a + N + shift_max]
        # 在检索区域中查找和x_tilde相似度最大的帧，保存偏移量
        if len(x_tilde) < N or len(retrieve_region) < 2 * N:
            break
        score_max = 0 # 最大相似度（使用内积）
        search_len = (2 * shift_max) if k + H_a - shift_max >= 0 else shift_max
        for j in range(search_len):
            score = np.dot(np.array(x_tilde).reshape(1, N), np.array(retrieve_region[j: j + N].reshape(N, 1)))
            if score > score_max:
                score_max = score
                shift = j - shift_max
        output[i * H_s : i * H_s + N] += y_buffer
        shift_list.append(shift)
        score_list.append(int(score_max) / N)
        # 取下一帧数据
        k += H_a
 
    # 保存最终的输出文件
    wavfile.write("./out.wav", freq, output.astype(np.int16))
    plt.plot(shift_list, label='shift')
    plt.plot(score_list, label='score')
    plt.legend()
    plt.show()