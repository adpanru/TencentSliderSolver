import os
import time
import json
import requests
from PIL import Image
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def download_image(url, output_path):
    """
    下载图片并保存到指定路径
    """
    try:
        # 添加模拟浏览器的headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://turing.captcha.qcloud.com/',
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # 确认返回的是图片内容
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"警告: 服务器返回的不是图片 (Content-Type: {content_type})")
            # 尝试保存内容，即使不是图片

        # 保存原始图片
        with open(output_path, 'wb') as f:
            f.write(response.content)

        # 验证图片有效性
        try:
            with Image.open(output_path) as img:
                width, height = img.size
                print(f"图片已保存到: {output_path}，尺寸: {width}x{height}")
                return True
        except Exception as e:
            print(f"保存的文件不是有效的图片: {str(e)}")
            # 返回错误内容以供调试
            with open(f"{output_path}.response.txt", 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"状态码: {response.status_code}\n")
                f.write(f"请求头: {str(response.request.headers)}\n")
                f.write(f"响应头: {str(response.headers)}\n")
                f.write(f"内容前100字节: {str(response.content[:100])}\n")
            return False

    except Exception as e:
        print(f"下载图片时出错: {str(e)}")
        return False


def process_slider_image(input_path, output_path, bg_size_width=345.06, bg_size_height=313.69, bg_pos_x=-70.8333,
                         bg_pos_y=-247.917, slider_width=60.7143, slider_height=60.7143):
    """
    修复滑块图片处理，使用正确的裁剪区域

    在CSS中，滑块元素的规格为：
    width: 60.7143px; height: 60.7143px;
    background-position: -70.8333px -247.917px;
    background-size: 345.06px 313.69px;
    """
    try:
        # 读取原始滑块图片
        img = Image.open(input_path)
        print(f"原始滑块图片尺寸: {img.size}")

        # 计算原始图像缩放比例
        original_width, original_height = img.size
        target_width, target_height = bg_size_width, bg_size_height

        # 先按照背景尺寸缩放整个图片
        scaled_img = img.resize((int(target_width), int(target_height)))
        print(f"缩放后尺寸: {scaled_img.size}")

        # 在CSS中，background-position负值表示图片向左/向上偏移
        # 即滑块元素中显示的是背景图片中从(bg_x, bg_y)开始的部分
        bg_x, bg_y = -bg_pos_x, -bg_pos_y  # 取正值，因为CSS中是负值

        # 确保裁剪区域在图片范围内
        if bg_x + slider_width > target_width:
            print(f"警告: X坐标超出范围，调整为图片宽度")
            bg_x = target_width - slider_width

        if bg_y + slider_height > target_height:
            print(f"警告: Y坐标超出范围，调整为图片高度")
            bg_y = target_height - slider_height

        # 正确裁剪区域
        crop_region = (int(bg_x), int(bg_y), int(bg_x + slider_width), int(bg_y + slider_height))
        print(f"裁剪区域: {crop_region}")

        # 裁剪出滑块图片
        cropped_img = scaled_img.crop(crop_region)
        print(f"裁剪后尺寸: {cropped_img.size}")

        # 保存处理后的图片
        cropped_img.save(output_path, format='PNG')
        print(f"修复后的滑块图片已保存到: {output_path}")

        return output_path
    except Exception as e:
        print(f"处理滑块图片时出错: {str(e)}")
        return None


def process_bg_preserve_ratio(input_path, output_path, target_width=340, target_height=242.857):
    """
    根据CSS参数处理背景图片，由于CSS指定background-size: 100%，直接调整尺寸
    """
    try:
        # 读取图片
        img = Image.open(input_path)
        print(f"原始图片尺寸: {img.size}")

        # 直接调整图片尺寸以适应目标尺寸，因为CSS指定background-size: 100%
        resized_img = img.resize((int(target_width), int(target_height)))
        print(f"调整后尺寸: {target_width}x{target_height}")

        # 保存处理后的图片
        resized_img.save(output_path, format='PNG')
        print(f"处理后的图片已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return False


def load_images(bg_path, slider_path):
    """加载背景图和滑块图片"""
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f"无法加载背景图片: {bg_path}")
        return None, None

    slider_img = cv2.imread(slider_path, cv2.IMREAD_UNCHANGED)
    if slider_img is None:
        print(f"无法加载滑块图片: {slider_path}")
        return None, None

    print(f"背景图片尺寸: {bg_img.shape}")
    print(f"滑块图片尺寸: {slider_img.shape}")

    return bg_img, slider_img


def preprocess_slider(slider_img, output_dir=None):
    """预处理滑块图片，提取滑块轮廓"""
    # 转为灰度图
    slider_gray = cv2.cvtColor(slider_img, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(slider_gray, 50, 150)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "slider_edges_fixed.png"), edges)
    else:
        cv2.imwrite("slider_edges_fixed.png", edges)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建掩码
    mask = np.zeros_like(slider_gray)
    cv2.drawContours(mask, contours, -1, 255, 2)

    # 膨胀操作，使边缘更明显
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "slider_mask_fixed.png"), dilated_mask)
    else:
        cv2.imwrite("slider_mask_fixed.png", dilated_mask)

    return dilated_mask


def parse_css_from_file(file_path):
    """
    从文件中读取CSS文本并解析出URL和CSS属性

    参数:
        file_path (str): CSS文件路径

    返回:
        dict: 包含背景图和滑块图信息的字典
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            css_lines = f.readlines()

        print(f"从文件 {file_path} 读取了 {len(css_lines)} 行CSS")

        return parse_captcha_params(css_lines)

    except Exception as e:
        print(f"解析CSS文件时出错: {str(e)}")
        return None


def parse_captcha_params(css_lines):
    """
    从CSS文本行列表中解析出URL和CSS属性

    参数:
        css_lines (list): CSS文本行列表，第一行为背景图CSS，第二行为滑块CSS

    返回:
        dict: 包含背景图和滑块图信息的字典
    """
    try:
        # 初始化结果字典
        result = {
            "bg": {
                "url": "",
                "width": 0,
                "height": 0,
                "position": {"top": 0, "left": 0},
                "background-size": "100%",
                "background-position": {"x": 0, "y": 0}
            },
            "slider": {
                "url": "",
                "width": 0,
                "height": 0,
                "position": {"top": 0, "left": 0},
                "background-size": {"width": 0, "height": 0},
                "background-position": {"x": 0, "y": 0}
            }
        }

        # 解析背景图CSS (第一行)
        if len(css_lines) > 0:
            bg_css = css_lines[0]

            # 提取宽度和高度
            width_match = bg_css.find("width: ")
            if width_match != -1:
                width_str = bg_css[width_match + 7:].split("px")[0]
                result["bg"]["width"] = float(width_str)

            height_match = bg_css.find("height: ")
            if height_match != -1:
                height_str = bg_css[height_match + 8:].split("px")[0]
                result["bg"]["height"] = float(height_str)

            # 提取背景图URL
            bg_url_match = bg_css.find("background-image: url(&quot;")
            if bg_url_match != -1:
                bg_url_end = bg_css.find("&quot;)", bg_url_match)
                bg_url = bg_css[bg_url_match + 27:bg_url_end]
                # 移除URL前面可能的分号
                if bg_url.startswith(";"):
                    bg_url = bg_url[1:]
                result["bg"]["url"] = bg_url

        # 解析滑块CSS (第二行)
        if len(css_lines) > 1:
            slider_css = css_lines[1]

            # 提取滑块URL
            slider_url_match = slider_css.find("background-image: url(&quot;")
            if slider_url_match != -1:
                slider_url_end = slider_css.find("&quot;)", slider_url_match)
                slider_url = slider_css[slider_url_match + 27:slider_url_end]
                # 移除URL前面可能的分号
                if slider_url.startswith(";"):
                    slider_url = slider_url[1:]
                result["slider"]["url"] = slider_url

            # 提取滑块宽度和高度
            width_match = slider_css.find("width: ")
            if width_match != -1:
                width_str = slider_css[width_match + 7:].split("px")[0]
                result["slider"]["width"] = float(width_str)

            height_match = slider_css.find("height: ")
            if height_match != -1:
                height_str = slider_css[height_match + 8:].split("px")[0]
                result["slider"]["height"] = float(height_str)

            # 提取滑块位置
            left_match = slider_css.find("left: ")
            if left_match != -1:
                left_str = slider_css[left_match + 6:].split("px")[0]
                result["slider"]["position"]["left"] = float(left_str)

            top_match = slider_css.find("top: ")
            if top_match != -1:
                top_str = slider_css[top_match + 5:].split("px")[0]
                result["slider"]["position"]["top"] = float(top_str)

            # 提取背景尺寸
            bg_size_match = slider_css.find("background-size: ")
            if bg_size_match != -1:
                bg_size_str = slider_css[bg_size_match + 17:].split(";")[0]
                bg_size_parts = bg_size_str.split()
                if len(bg_size_parts) >= 2:
                    result["slider"]["background-size"]["width"] = float(bg_size_parts[0].replace("px", ""))
                    result["slider"]["background-size"]["height"] = float(bg_size_parts[1].replace("px", ""))

            # 提取背景位置
            bg_pos_match = slider_css.find("background-position: ")
            if bg_pos_match != -1:
                bg_pos_str = slider_css[bg_pos_match + 21:].split(";")[0]
                bg_pos_parts = bg_pos_str.split()
                if len(bg_pos_parts) >= 2:
                    result["slider"]["background-position"]["x"] = float(bg_pos_parts[0].replace("px", ""))
                    result["slider"]["background-position"]["y"] = float(bg_pos_parts[1].replace("px", ""))

        print("CSS解析结果:")
        print(json.dumps(result, indent=4))

        return result

    except Exception as e:
        print(f"解析CSS数据时出错: {str(e)}")
        return None


def match_template(bg_img, slider_mask, output_dir=None):
    """模板匹配，查找滑块在背景图中的位置"""
    methods = [
        ('cv2.TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('cv2.TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('cv2.TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
    ]

    results = []

    # 转换背景为灰度图
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    bg_edges = cv2.Canny(bg_gray, 50, 150)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "bg_edges_fixed.png"), bg_edges)
    else:
        cv2.imwrite("bg_edges_fixed.png", bg_edges)

    for method_name, method in methods:
        print(f"使用方法: {method_name}")

        # 使用边缘进行匹配
        result = cv2.matchTemplate(bg_edges, slider_mask, method)

        if method == cv2.TM_SQDIFF_NORMED:
            # 对于SQDIFF方法，值越小表示匹配度越高
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            confidence = 1 - min_val  # 转换为置信度
            match_loc = min_loc
            print(f"  最小值: {min_val:.4f}, 最大值: {max_val:.4f}")
            print(f"  最佳匹配位置: {min_loc}, 置信度: {confidence:.4f}")
        else:
            # 对于其他方法，值越大表示匹配度越高
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            confidence = max_val
            match_loc = max_loc
            print(f"  最小值: {min_val:.4f}, 最大值: {max_val:.4f}")
            print(f"  最佳匹配位置: {max_loc}, 置信度: {confidence:.4f}")

        results.append((method_name, match_loc, confidence, result))

    # 找到置信度最高的结果
    best_result = max(results, key=lambda x: x[2])

    print(f"\n最佳匹配结果: {best_result[0]}")
    print(f"位置: {best_result[1]}, 置信度: {best_result[2]:.4f}")

    return best_result


def visualize_result(bg_img, slider_img, method_name, match_loc, result_matrix, slider_initial_left, output_dir=None):
    """可视化匹配结果"""
    h, w = slider_img.shape[:2]

    # 在匹配位置绘制矩形
    result_img = bg_img.copy()
    x, y = match_loc
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 同时显示初始位置
    initial_x = int(slider_initial_left)
    cv2.rectangle(result_img, (initial_x, y), (initial_x + w, y + h), (0, 255, 0), 2)

    # 绘制从初始位置到匹配位置的箭头
    cv2.arrowedLine(result_img, (initial_x + w // 2, y + h // 2), (x + w // 2, y + h // 2),
                    (255, 0, 0), 2, tipLength=0.3)

    # 标注偏移距离
    offset = x - slider_initial_left
    cv2.putText(result_img, f"Offset: {offset:.2f} px", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 保存匹配结果图片
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "match_result_with_offset.png"), result_img)
    else:
        cv2.imwrite("match_result_with_offset.png", result_img)

    # 绘制热力图
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
    plt.title("Background Image")

    plt.subplot(132)
    plt.imshow(slider_img, cmap='gray')
    plt.title("Slider Image")

    plt.subplot(133)
    plt.imshow(result_matrix, cmap='jet')
    plt.colorbar()
    plt.title(f"Match Result ({method_name})")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "match_heatmap_with_offset.png"))
    else:
        plt.savefig("match_heatmap_with_offset.png")
    plt.close()


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='验证码滑块距离计算工具')
    parser.add_argument('--bg-css', type=str, help='背景图CSS字符串')
    parser.add_argument('--slider-css', type=str, help='滑块CSS字符串')
    parser.add_argument('--css-file', type=str, help='CSS文件路径，文件包含两行：第一行是背景图CSS，第二行是滑块CSS')

    # 解析命令行参数
    args = parser.parse_args()

    # 硬编码指定输出目录  todo 改为自己要保存图片的位置
    output_dir = r"\image"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 时间戳作为文件名前缀
    timestamp = int(time.time())

    # 处理命令行参数
    css_params = None

    if args.bg_css and args.slider_css:
        print("使用命令行提供的CSS字符串")
        css_lines = [args.bg_css, args.slider_css]
        css_params = parse_captcha_params(css_lines)
    elif args.css_file:
        print(f"使用指定的CSS文件: {args.css_file}")
        if os.path.exists(args.css_file):
            css_params = parse_css_from_file(args.css_file)
        else:
            print(f"指定的CSS文件不存在: {args.css_file}")
    else:
        # 尝试从xiaoETongUrls文件解析CSS
        css_file_path = os.path.join(".", "xiaoETongUrls")
        if os.path.exists(css_file_path):
            print(f"发现CSS文件: {css_file_path}")
            css_params = parse_css_from_file(css_file_path)
        else:
            print(f"未找到xiaoETongUrls文件，请确认文件存在")
            sys.exit(1)

    if not css_params:
        print("无法解析CSS参数，请检查文件格式是否正确")
        sys.exit(1)

    # 使用从CSS文件解析的参数
    prefix = f"captcha_top_{css_params['slider']['position']['top']}_{timestamp}"

    # 背景图片URL
    bg_url = css_params["bg"]["url"]

    # 滑块图片URL
    slider_url = css_params["slider"]["url"]

    # 保存CSS参数
    params_path = os.path.join(output_dir, f"{prefix}_params.json")
    with open(params_path, 'w') as f:
        json.dump(css_params, f, indent=4)
    print(f"CSS参数已保存到: {params_path}")

    # 设置路径
    bg_raw_path = os.path.join(output_dir, f"{prefix}_bg_raw.jpg")
    slider_raw_path = os.path.join(output_dir, f"{prefix}_slider_raw.jpg")
    processed_bg_path = os.path.join(output_dir, f"{prefix}_bg_preserved_ratio.png")
    processed_slider_path = os.path.join(output_dir, f"{prefix}_slider_fixed.png")

    # 下载和处理图片的标志
    bg_processed = False
    slider_processed = False

    # 下载背景图片
    if download_image(bg_url, bg_raw_path):
        # 处理背景图片 - 保持原始宽高比
        if process_bg_preserve_ratio(bg_raw_path, processed_bg_path, css_params["bg"]["width"],
                                     css_params["bg"]["height"]):
            bg_processed = True
        else:
            print("背景图片处理失败")
    else:
        print("背景图片下载失败")

    # 下载滑块图片
    if download_image(slider_url, slider_raw_path):
        # 处理滑块图片 - 使用从CSS提取的所有参数
        if process_slider_image(
                slider_raw_path,
                processed_slider_path,
                css_params["slider"]["background-size"]["width"],
                css_params["slider"]["background-size"]["height"],
                css_params["slider"]["background-position"]["x"],
                css_params["slider"]["background-position"]["y"],
                css_params["slider"]["width"],
                css_params["slider"]["height"]):
            slider_processed = True
        else:
            print("滑块图片处理失败")
    else:
        print("滑块图片下载失败")

    # 只有当两个图片都成功处理后才继续
    if bg_processed and slider_processed:
        # 加载图片并进行模板匹配
        bg_img, slider_img = load_images(processed_bg_path, processed_slider_path)

        # 检查图片是否成功加载
        if bg_img is not None and slider_img is not None:
            # 预处理滑块
            slider_mask = preprocess_slider(slider_img, output_dir)

            # 执行模板匹配
            method_name, match_loc, confidence, result_matrix = match_template(bg_img, slider_mask, output_dir)

            # 可视化结果
            visualize_result(bg_img, slider_mask, method_name, match_loc, result_matrix,
                             css_params["slider"]["position"]["left"], output_dir)

            # 计算滑块需要移动的距离（像素）
            x, y = match_loc
            actual_distance = x - css_params["slider"]["position"]["left"]
            print(f"\n滑块需要移动的距离: {x} 像素")
            print(f"考虑初始位置后的实际移动距离: {actual_distance} 像素")

            # 创建结果JSON文件
            result_data = {
                "timestamp": timestamp,
                "bg_url": bg_url,
                "slider_url": slider_url,
                "slider_position": {
                    "top": css_params["slider"]["position"]["top"],
                    "left": css_params["slider"]["position"]["left"]
                },
                "match_position": {
                    "x": int(match_loc[0]),
                    "y": int(match_loc[1])
                },
                "move_distance": actual_distance,
                "confidence": confidence
            }

            result_path = os.path.join(output_dir, f"{prefix}_result.json")
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            print(f"结果数据已保存到: {result_path}")
        else:
            print("图片加载失败，无法进行模板匹配")
    else:
        print("由于图片下载或处理失败，跳过模板匹配步骤")


if __name__ == "__main__":
    main() 