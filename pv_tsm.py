# -*- coding: utf-8 -*-
# @Time    : 2021-05-07
# @Author  : Zhangjieyang
# @Email   : jieyangzhang@outlook.com
# @File    : pv_tsm.py
# @Software: vscode
 
# 数学计算库
import numpy as np
# 读写音频会用到的库
from scipy.io import wavfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import click
 
@click.command()
@click.option(
    '--scaling',
    prompt='please input scaling',
    type=float,
    default='1',
    help='Time Scaling'
)
 
def main(scaling):
    # 读取音频文件
    freq, sig = wavfile.read("./audio.wav")
    len_data = len(sig)
    # -----------可配置项-start----------
    alpha = scaling # 时长调整比例 alpha = H_s / H_a
    # -----------可配置项--end-----------
 
    # ----------默认配置项-start---------
    rate = freq
    N = int(100 / 1000 * rate)  # 50ms时长的音频帧长
    nfft = int(N / 2 + 1)    # 傅立叶有效点数
    H_s = int(N / 32)           # 输出帧间隔
    H_a = int(H_s / alpha)     # 原始信号分帧间隔
    win = np.hanning(N) # 窗函数
    print('window size : %d, syn hop size : %d, analyze hop size : %d' % (N, H_s, H_a))
    imstft_norm = np.zeros([N])
    for l in range(N):
        imstft_norm[l] += (win[l] ** 2)
        z = -1
        while 1:
            if (l + z * H_s) < 0:
                break
            imstft_norm[l] += (win[l + z * H_s] ** 2)
            z = z - 1
        z = 1
        while 1:
            if (l + z * H_s) >= N:
                break
            imstft_norm[l] += (win[l + z * H_s] ** 2)
            z = z + 1
    # ----------默认配置项--end----------
    num_frames = int((len_data - N + H_a) / H_a) # 分帧数量
    freq_resolution = freq / N
    phase_resolution = freq_resolution * H_a / freq * 2 * np.pi
    phase_tmp = np.zeros([N]) # 上一帧的真实相位信息
    phase_mod = np.zeros([N]) # 上一帧的调整相位信息
    print('freq resolution : %d, phase resolution : %d' % (freq_resolution, phase_resolution))
 
    # ------初始化-start---------
    output = np.zeros([(num_frames - 1) * H_s + N])   # 输出信号
    # ------初始化--end----------
    x_buff = sig[0 : N] * win # 第一帧
    X_buff = np.fft.rfft(x_buff, N)
    phase_tmp = np.arctan2(X_buff.imag, X_buff.real).copy()
    phase_mod = np.arctan2(X_buff.imag, X_buff.real).copy()
    output[0 : N] = x_buff
 
    # 循环体，从第二帧开始进行处理
    for i in tqdm(range(1, num_frames)):
        # 分帧
        x_buff = sig[i * H_a : i * H_a + N] * win # 当前需要处理的帧
        if len(x_buff) < N:
            break
        # 傅立叶变换
        X_buff = np.fft.rfft(x_buff, N)
        magnitude = abs(X_buff)
        phase = np.arctan2(X_buff.imag, X_buff.real)
 
        coarse_freq_list = [] # 频带对应的频率
        phase_sub_list = [] # 相位差值
        current_freq_list = []
        last_freq_list = []
        estimate_freq_list = []
        err_phase_list = []
 
        # 瞬时频率预测
        for f in range(nfft):
            coarse_freq = f * rate / N # stft得到的
            phase_pred = phase_tmp[f] + H_a / rate * (coarse_freq * 2 * np.pi)
            phase_tmp[f] = phase[f].copy()
            phase_err = phase[f] - phase_pred
            # 将phase_err映射到[-0.5, 0.5]*频谱分辨率对应的相位范围
            phase_err = phase_err % (2 * np.pi)
            if phase_err > np.pi:
                phase_err = phase_err - 2 * np.pi
            if phase_err > 0.5 * phase_resolution:
                phase_err = 0
            if phase_err < -0.5 * phase_resolution:
                phase_err = 0
            estimage_omega = coarse_freq * np.pi * 2 + phase_err / (H_a / rate)
            estimate_freq = estimage_omega / (2 * np.pi)
            show = magnitude[f] / np.max(magnitude) > 0
            if phase_err != 0 and show:
                # print('estimate freq:%f, i:%d, f:%d, magnitude:%f, coarse_freq:%f, phase err:%f, phase pred:%f' %
                #     (estimate_freq, i, f, magnitude[f], coarse_freq, phase_err, phase_pred))
                phase[f] = phase_mod[f] + H_s / rate * estimage_omega
                phase[f] = phase[f] % (2 * np.pi)
                if phase[f] > np.pi:
                    phase[f] = phase[f] - 2 * np.pi
        X_buff = magnitude * np.exp(phase*1j)
        # 傅立叶逆变换
        phase_mod = phase.copy()
        segment = np.fft.irfft(X_buff) * win
        segment = segment / imstft_norm
        # 迭接相加法
        output[i * H_s : i * H_s + N] += segment
        # 取下一帧数据
 
    # 保存最终的输出文件
    wavfile.write("./out.wav", freq, output.astype(np.int16))
 
if __name__ == "__main__":
    main()