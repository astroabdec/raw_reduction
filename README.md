ASTROABDEC 数据预处理小组
============================

原始数据下载地址：
* [PFS data](https://kdocs.cn/l/ckfHYmmkGJYq), over 22 GB
* [APF data](https://kdocs.cn/l/cjxKJIT6zlxw)
* [观测日志](https://kdocs.cn/l/cnay2iUPKhSc)

数据预处理流程笔记：
* [数据处理 step by step](https://github.com/astroabdec/raw_reduction/blob/main/ABDEC2024_raw_reduction.ipynb)

第一天
-------
**任务1：display the spectrum**
* 理解和读入logsheets
* 读入光谱
* 分辨出红端和蓝端，波长增长的方向，overscan region
* optional: 写一个python程序，读入和旋转frame，波长从左往右增加，从frame下端往上增加，这样spec[i,j] is always bluer than spec[i+N, j+N]

**任务2：bias subtraction**
* APF rotate之后最下面32个rows是overscan，用这个部分做bias subtraction
* PFS的bias frames做median combine，得到master bias
* 画一些bias subtraction之前和之后的图，cutting across along the cross-dispersion direction

第二天/第三天
--------------------
用PFS的数据练手（因为后续杂散光更容易做）

**任务1：做一个master flat**
* 所有flat frames减bias
* median combine all flat frames to make a master flat
* Wide和Narrow应该各有一个master flat

**任务2：trace out the orders**
* order positions determined by Paul Butler https://kdocs.cn/l/clVIBGLVTS2W
* order positions determined by Liang Wang https://kdocs.cn/l/col0MtT5YZvP

第四天
---------
Goal 1: Flat Fielding
Goal 2: Scattered light removal

第五天
----------


波长定标所需的程序和文件：
* [波长证认程序](https://github.com/astroabdec/raw_reduction/blob/main/ident_wave.py)
* [ThAr谱线波长列表](https://github.com/astroabdec/raw_reduction/blob/main/thar.dat)
* [兴隆观测基地HRS波长证认图](https://github.com/astroabdec/raw_reduction/blob/main/Xinglong216%20-%20HRS.pdf)
* [Keck HIRES蓝端ThAr波长证认图](https://github.com/astroabdec/raw_reduction/blob/main/Keck%20-%20HIRES_1CCD_b.pdf)

用法： 在终端里运行

```bash
./ident_wave.py thar.fits
```

在弹出的图形界面上对照兴隆HRS的灯谱证认图证认波长。紫外部分对照Keck的紫外ThAr灯谱图证认。关闭后在工作目录里生成 `wlcalib_thar.fits` 文件。各个级次波长解在第二个HDU里以二进制表格形式存储。
