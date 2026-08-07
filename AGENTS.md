## 编码与中文输出

- 源文件统一使用 UTF-8（无 BOM）保存。
- 使用 `json.dumps` 在终端显示结构化结果时，设置 `ensure_ascii=False`，避免中文显示为 `\uXXXX` 转义序列：

```python
formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
```
