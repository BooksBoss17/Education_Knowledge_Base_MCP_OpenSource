# 极端长宽比模型输入补白

开发版已接入此前冻结验证过的无损补白机制。与单色、双色排除共用 `src/bemarkdown/blank_images.py` 模块，函数为 `prepare_model_image`；排除函数 `blank_reason`、`prune_docx`、`prune_pdf` 保持原逻辑。

单色、双色排除发生在输出整理阶段，补白发生在模型读图之前。实际接入点为独立 OvisOCR2 / Xiaomi-OCR-0 工作进程 `pdf/workers/qwen_ocr.py`，该输入用于整图纯文转录，未改源页坐标或交付图片。它不是按长宽比删图，也不替代冻结的 B200 筛选方案。

- 仅 `max(width,height)/min(width,height)>200` 时触发；短边补至至少32像素，并保证补后比例不超过200。
- 白边为不透明 RGBA；原图区域居中、原RGBA像素逐值保留，无缩放、裁切或原文件覆盖。普通图片沿用原模式、尺寸和像素。
- 派生图最多4千万像素，超限在分配之前拒绝；动画也不静默丢弃帧。工作进程将失败记录为识别失败，不由补白函数产生删除结论。
- 模型缓存指纹包含补白模块SHA和固定参数；每个结果记录变换信息和实际RGB输入SHA，源裁图SHA仍指向原文件。

验证：22项CPU测试通过，包括工作进程实际调用路径的模型替身、缓存命中/辅助源码变化失效、失败记录、普通图片边界、RGBA像素保护与原单色/双色规则。另以原冻结开发集的8项补白输入做逐像素SHA比对，派生图与冻结方案一致，原文件SHA不变。未加载权重、未执行GPU推理，不作新增OCR质量或完整流水线内存认证。

```powershell
Set-Location '<BeMarkdown-source>'
& '.venv/Scripts/python.exe' -m pytest tests/test_extreme_aspect_model_input.py tests/test_blank_images.py tests/test_fragment_and_two_color_policy.py tests/test_low_vram_worker_image_budget.py tests/test_production_worker_runtime.py
```

验证和修改前源文件保存在仓库 `tmp/bemarkdown-aspect-padding-20261007`。当前仅修改开发区 BeMarkdown 源码，尚未同步到正式MCP安装目录。
