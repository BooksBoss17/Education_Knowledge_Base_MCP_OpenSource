# 独立图片输入

文本PDF、扫描PDF、原生Word、截图Word保持现有处理路线。独立图片现在也可通过同一生产入口转换：

```powershell
python -m bemarkdown convert "C:\input\page.png" --output-root "C:\workspace\tmp\bemarkdown" --debug --json
```

支持PNG、JPEG（.jpg/.jpeg）、WebP、BMP、TIFF（.tif/.tiff，单帧），大小上限128 MiB、4000万像素。内容损坏、扩展名与格式不符、多帧图片在加载模型前失败；不支持SVG、HEIC、GIF或图片目录隐式合并。多张图片可显式传给convert-batch，每个文件一个独立包。

输入按EXIF方向旋正，透明度合成到白底，不缩放原图像素。以216 DPI建立单页PDF（每个像素对应1/3 PDF point），PNG无损嵌入；复用现有布局、文字、公式、表格及示意图流程，没有增加识别模型。派生PDF内的图片像素保持归一化结果，后续布局/OCR的渲染仍由原有流程控制。

package_manifest与conversion_report中的source为IMAGE及原文件SHA；input_transform.route为RASTER_IMAGE_PDF，记录原始/归一化尺寸、EXIF、归一化PNG SHA、派生PDF SHA及坐标比例。DocumentIR及OCR候选仍描述派生PDF坐标，不冒充原始图片坐标；debug/source-pagination.pdf是可校验的坐标载体。

MCP开发适配器的bemarkdown_convert接受图片路径并自动要求debug；bemarkdown_source可读取旋正/白底后的原图或使用source_view=pagination读取哈希绑定派生页。region均为对应视图的归一化坐标。直接原图视图不改变像素密度，dpi仅控制派生PDF渲染；原图没有原生文本层，kind=text不会把OCR伪装成原始文本。

原有bemarkdown_convert_image(job_id,asset_name)保留，继续处理父任务资产。独立图片使用bemarkdown_convert(source)，无需先造一个父任务。

入口运行成功不代表内容已经审校通过。待识别公式、表格和边界问题沿用原来的待审机制。该入口的发布必须同步更新BeMarkdown wheel与适配器；实际安装身份以正式MCP的发布清单及运行时校验为准。
