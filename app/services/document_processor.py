import hashlib
import io
from pathlib import Path
from typing import List, Tuple, Optional


class DocumentParser:
    """文档解析器 - 把上传的文件变成纯文字"""

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
        """
        解析文档
        返回: (文字内容, 错误信息)
        """
        ext = Path(filename).suffix.lower()

        try:
            if ext == ".pdf":
                return DocumentParser._parse_pdf(file_bytes)
            elif ext == ".docx":
                return DocumentParser._parse_docx(file_bytes)
            elif ext in (".txt", ".md"):
                return DocumentParser._parse_text(file_bytes)
            else:
                return "", f"不支持的格式: {ext}"
        except Exception as e:
            return "", f"解析失败: {e}"

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """解析 PDF (优先 PyMuPDF，备用 pdfminer)"""
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(texts), None
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            return extract_text(io.BytesIO(file_bytes)), None
        except ImportError:
            return "", "缺少 PDF 解析库，请运行: pip install PyMuPDF pdfminer.six"

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """解析 Word 文档"""
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(texts), None

    @staticmethod
    def _parse_text(file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """解析纯文本"""
        for encoding in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                return file_bytes.decode(encoding), None
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace"), None


class TextSplitter:
    """智能文字分块器 - 把大段文字切成小块"""

    @staticmethod
    def split(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将文字切成小块
        策略:
        1. 按自然段落分
        2. 太长的段落再强行切分
        3. 块与块之间保留一些重复内容(overlap)
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果加上这一段还没超出限制
            if len(current) + len(para) <= chunk_size:
                current += para + "\n\n"
            else:
                # 保存当前块
                if len(current.strip()) >= 30:
                    chunks.append(current.strip())

                # 处理超长段落
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size - overlap):
                        sub = para[i:i + chunk_size]
                        if len(sub.strip()) >= 30:
                            chunks.append(sub.strip())
                    current = ""
                else:
                    # 保留上一块的结尾作为重叠
                    if overlap > 0 and current.strip():
                        tail = current.strip()[-overlap:]
                        current = tail + "\n\n" + para + "\n\n"
                    else:
                        current = para + "\n\n"

        # 最后一块
        if len(current.strip()) >= 30:
            chunks.append(current.strip())

        return chunks


def process_document(file_bytes: bytes, filename: str):
    """
    完整处理流程: 解析 → 分块 → 返回文字和块
    """
    text, error = DocumentParser.parse(file_bytes, filename)
    if error:
        raise ValueError(error)

    chunks = TextSplitter.split(text)
    return text, chunks
