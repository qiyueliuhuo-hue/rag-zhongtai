import fitz  # PyMuPDF
from docx import Document  # python-docx

# from langchain_community.document_loaders import UnstructuredWordDocumentLoader

import os


def _parse_pdf(file_path):
    """PDF：逐页提取文本"""
    doc = fitz.open(file_path)
    text_list = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_list.append(text.strip())
    doc.close()
    return text_list


def _parse_docx(file_path):
    """Word：逐段落提取文本"""
    doc = Document(file_path)
    text_list = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            text_list.append(text)
    return text_list


def _parse_text(file_path):
    """MD/TXT：直接读文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]


print("开始加载文档...")
LOADER_REGISTRY = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".doc": _parse_docx,
    ".md": _parse_text,
    ".txt": _parse_text,
}


def load_document(file_path):
    """
    根据文件后缀自动选择合适的加载器加载文档
    """
    # 获取文件后缀
    file_extension = os.path.splitext(file_path)[1].lower()
    # 根据后缀选择加载器
    loader = LOADER_REGISTRY.get(file_extension)
    if loader is None:
        raise ValueError(f"Unsupported file format: {file_extension}")
    # 加载文档
    try:
        text_list = loader(file_path)
        return text_list
    except Exception as e:
        raise ValueError(f"Error loading document: {e}")


print("加载完成")
