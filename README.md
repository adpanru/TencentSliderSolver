# 腾讯滑块验证码距离计算工具

这个工具能够自动分析腾讯滑块验证码，计算滑块需要移动的精确距离。通过分析验证码的CSS样式和图片内容，使用计算机视觉技术找出滑块的目标位置。

## 功能特点

- 自动解析验证码CSS样式信息
- 下载并处理背景图和滑块图片
- 使用多种模板匹配算法寻找最佳匹配位置
- 生成可视化结果，直观展示匹配效果
- 计算精确的滑块移动距离
- 输出详细的JSON结果

## 安装依赖

本工具需要以下Python库：

```bash
pip install requests pillow opencv-python numpy matplotlib
```

## 使用方法

### 方法一：使用CSS文件

1. 准备一个包含验证码CSS样式的文件（例如`xiaoETongUrls`）：
   - 第一行：背景图的CSS样式
   - 第二行：滑块的CSS样式

2. 运行脚本：

```bash
python process_current_captcha.py
```

脚本会自动查找当前目录下的`xiaoETongUrls`文件，并分析其中的CSS样式。

### 方法二：使用命令行参数

也可以通过命令行参数直接提供CSS样式：

```bash
python process_current_captcha.py --bg-css "背景图CSS样式" --slider-css "滑块CSS样式"
```

或者指定一个自定义的CSS文件：

```bash
python process_current_captcha.py --css-file "你的CSS文件路径"
```

## 输出结果

脚本会在`image`目录下生成以下文件：

- `captcha_top_XXX_TIMESTAMP_params.json`：解析出的CSS参数
- `captcha_top_XXX_TIMESTAMP_bg_raw.jpg`：原始背景图
- `captcha_top_XXX_TIMESTAMP_bg_preserved_ratio.png`：处理后的背景图
- `captcha_top_XXX_TIMESTAMP_slider_raw.jpg`：原始滑块图
- `captcha_top_XXX_TIMESTAMP_slider_fixed.png`：处理后的滑块图
- `bg_edges_fixed.png`：背景图边缘检测结果
- `slider_edges_fixed.png`：滑块边缘检测结果
- `slider_mask_fixed.png`：滑块掩码
- `match_result_with_offset.png`：匹配结果可视化
- `match_heatmap_with_offset.png`：匹配热力图
- `captcha_top_XXX_TIMESTAMP_result.json`：最终结果，包含滑块移动距离

## 实现原理

### 1. CSS解析

脚本首先解析验证码的CSS样式，提取出以下关键信息：

- 背景图和滑块图的URL
- 背景图尺寸：width和height
- 滑块尺寸和位置：width、height、top、left
- 滑块的background-size和background-position

### 2. 图片处理

根据CSS参数处理图片：

- 背景图：根据CSS中指定的尺寸进行缩放
- 滑块图：根据background-position和background-size从原始图片中裁剪出实际的滑块部分

### 3. 模板匹配

使用OpenCV的模板匹配算法寻找滑块在背景图中的目标位置：

1. 将背景图和滑块图转换为灰度图
2. 对两张图片进行边缘检测（Canny算法）
3. 使用三种不同的模板匹配算法进行匹配：
   - CV2.TM_CCOEFF_NORMED：归一化相关系数匹配
   - CV2.TM_CCORR_NORMED：归一化相关匹配
   - CV2.TM_SQDIFF_NORMED：归一化平方差匹配
4. 比较三种算法的置信度，选择最佳匹配结果

### 4. 距离计算

计算滑块需要移动的距离：

```
移动距离 = 匹配位置的X坐标 - 滑块初始位置的X坐标
```

### 5. 结果可视化

生成可视化结果，直观展示匹配效果：

- 在背景图上标记出滑块的初始位置（绿色框）和目标位置（红色框）
- 用箭头连接初始位置和目标位置
- 标注实际的移动距离
- 生成匹配算法的热力图

## 实际应用

本工具可以用于：

1. 自动化测试滑块验证码
2. 研究验证码的安全性和有效性
3. 学习计算机视觉和图像处理技术

## 注意事项

- 本工具仅供学习和研究使用
- 请勿用于任何非法活动或绕过正常的验证流程
- 在使用第三方API和网站时，请遵守相关服务条款 