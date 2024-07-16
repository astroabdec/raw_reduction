ASTROABDEC 数据预处理小组
============================

原始数据下载地址：
* [PFS data](https://wltransfer.s3.cn-north-1.amazonaws.com.cn/pfs.tar.gz), over 22 GB
* [APF data](https://wltransfer.s3.cn-north-1.amazonaws.com.cn/apf.tar.gz)
* [观测日志](https://kdocs.cn/l/cnay2iUPKhSc)


数据预处理笔记：
* [数据处理 step by step](https://github.com/astroabdec/raw_reduction/blob/main/ABDEC2024_raw_reduction.ipynb)

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
