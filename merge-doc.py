#!/usr/bin/env python3
"""
merge_docs.py

将 MkDocs 项目中的所有 Markdown 文件按 mkdocs.yml 中定义的导航顺序合并为一个文件。
输出:
    - merged-doc.md
"""

import pathlib
import sys

try:
    import yaml
except ImportError:
    print("需要安装 PyYAML: pip install pyyaml")
    sys.exit(1)


# 忽略 MkDocs 配置中的 Python 对象标签
class IgnorePythonNameLoader(yaml.SafeLoader):
    pass


def ignore_python_name(loader, node):
    # 对于 !!python/name:xxx 标签，直接返回一个占位字符串
    return "python_object"


IgnorePythonNameLoader.add_constructor(
    "tag:yaml.org,2002:python/name",
    ignore_python_name,
)

# 可能还有其他 python/object 标签，同样处理
IgnorePythonNameLoader.add_constructor(
    "tag:yaml.org,2002:python/object",
    ignore_python_name,
)


def extract_nav_files(nav, results=None):
    """递归提取导航中的所有文件路径。"""
    if results is None:
        results = []
    if isinstance(nav, list):
        for item in nav:
            extract_nav_files(item, results)
    elif isinstance(nav, dict):
        for key, value in nav.items():
            if isinstance(value, str):
                results.append(value)
            else:
                extract_nav_files(value, results)
    return results


def main():
    project_root = pathlib.Path(__file__).parent
    mkdocs_yml = project_root / "mkdocs.yml"
    docs_dir = project_root / "docs"

    if not mkdocs_yml.exists():
        print("错误: 未找到 mkdocs.yml")
        sys.exit(1)

    # 使用自定义加载器读取 mkdocs.yml
    with open(mkdocs_yml, encoding="utf-8") as f:
        config = yaml.load(f, Loader=IgnorePythonNameLoader)

    nav = config.get("nav", [])
    if not nav:
        print("错误: mkdocs.yml 中未定义 nav")
        sys.exit(1)

    file_paths = extract_nav_files(nav)
    print(f"在导航中找到 {len(file_paths)} 个文件。")

    merged_md = []
    missing_files = []

    for rel_path in file_paths:
        file_path = docs_dir / rel_path
        if not file_path.exists():
            missing_files.append(str(rel_path))
            continue

        text = file_path.read_text(encoding="utf-8")
        merged_md.append(text)
        print(f"  ✓ {rel_path}")

    if missing_files:
        print("\n警告: 以下文件不存在，已跳过:")
        for f in missing_files:
            print(f"  ✗ {f}")

    if not merged_md:
        print("\n错误: 没有成功合并任何文件。")
        sys.exit(1)

    merged_content = "\n\n---\n\n".join(merged_md)

    output_md = project_root / "merged-doc.md"
    output_md.write_text(merged_content, encoding="utf-8")
    print(f"\n已生成 Markdown 文件: {output_md}")


if __name__ == "__main__":
    main()