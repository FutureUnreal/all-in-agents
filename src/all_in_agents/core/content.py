from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    text: str

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImageUrlBlock:
    url: str
    detail: str = "auto"

    def to_dict(self) -> dict:
        return {"type": "image_url", "url": self.url, "detail": self.detail}


@dataclass(frozen=True)
class ImageBase64Block:
    data: str
    media_type: str = "image/jpeg"
    detail: str = "auto"

    def to_dict(self) -> dict:
        return {
            "type": "image_base64",
            "data": self.data,
            "media_type": self.media_type,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FileUrlBlock:
    url: str
    filename: str = ""
    media_type: str = "application/pdf"

    def to_dict(self) -> dict:
        return {
            "type": "file_url",
            "url": self.url,
            "filename": self.filename,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class FileBase64Block:
    data: str
    filename: str
    media_type: str = "application/pdf"

    def to_dict(self) -> dict:
        return {
            "type": "file_base64",
            "data": self.data,
            "filename": self.filename,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class FileIdBlock:
    file_id: str
    filename: str = ""
    media_type: str = "application/pdf"

    def to_dict(self) -> dict:
        return {
            "type": "file_id",
            "file_id": self.file_id,
            "filename": self.filename,
            "media_type": self.media_type,
        }


def text_block(text: str) -> dict:
    return TextBlock(text).to_dict()


def image_url_block(url: str, *, detail: str = "auto") -> dict:
    return ImageUrlBlock(url=url, detail=detail).to_dict()


def image_base64_block(data: str, *, media_type: str = "image/jpeg", detail: str = "auto") -> dict:
    return ImageBase64Block(data=data, media_type=media_type, detail=detail).to_dict()


def file_url_block(url: str, *, filename: str = "", media_type: str = "application/pdf") -> dict:
    return FileUrlBlock(url=url, filename=filename, media_type=media_type).to_dict()


def file_base64_block(data: str, *, filename: str, media_type: str = "application/pdf") -> dict:
    return FileBase64Block(data=data, filename=filename, media_type=media_type).to_dict()


def file_id_block(file_id: str, *, filename: str = "", media_type: str = "application/pdf") -> dict:
    return FileIdBlock(file_id=file_id, filename=filename, media_type=media_type).to_dict()


def normalize_content(content: Any) -> str | list[dict]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [normalize_content_block(block) for block in content]
    raise TypeError("message content must be a string or list of content blocks")


def normalize_content_block(block: Any) -> dict:
    if hasattr(block, "to_dict"):
        block = block.to_dict()
    if not isinstance(block, dict):
        raise TypeError("content blocks must be dicts or objects with to_dict()")

    btype = block.get("type")
    if btype == "input_text":
        return {"type": "text", "text": str(block.get("text", ""))}

    if btype == "text":
        return {"type": "text", "text": str(block.get("text", ""))}

    if btype in ("image_url", "input_image"):
        url, detail = _extract_image_url(block)
        if url:
            return {"type": "image_url", "url": url, "detail": detail}

    if btype == "image_base64":
        return {
            "type": "image_base64",
            "data": str(block.get("data", "")),
            "media_type": str(block.get("media_type", "image/jpeg") or "image/jpeg"),
            "detail": str(block.get("detail", "auto") or "auto"),
        }

    if btype in ("file_url", "input_file"):
        file_id = str(block.get("file_id", "") or "")
        file_data = str(block.get("file_data", "") or "")
        if btype == "input_file" and file_id:
            return _file_id_block(block, file_id=file_id)
        if btype == "input_file" and file_data:
            return _file_base64_block(block, data=file_data)
        return _file_url_block(block)

    if btype == "file_base64":
        return _file_base64_block(block, data=str(block.get("data", "") or ""))

    if btype == "file_id":
        return _file_id_block(block, file_id=str(block.get("file_id", "") or ""))

    if btype == "file" and isinstance(block.get("file"), dict):
        raw_file = block["file"]
        file_id = str(raw_file.get("file_id", "") or "")
        file_data = str(raw_file.get("file_data", "") or "")
        if file_id:
            return _file_id_block(raw_file, file_id=file_id)
        if file_data:
            return _file_base64_block(raw_file, data=file_data)

    return dict(block)


def is_image_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in {"image_url", "image_base64", "input_image"}


def is_file_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in {"file_url", "file_base64", "file_id", "input_file", "file"}


def image_url_for_provider(block: dict) -> str:
    btype = block.get("type")
    if btype == "image_url":
        return str(block.get("url", ""))
    if btype == "input_image":
        return str(block.get("image_url", ""))
    if btype == "image_base64":
        data = str(block.get("data", ""))
        if data.startswith("data:"):
            return data
        media_type = str(block.get("media_type", "image/jpeg") or "image/jpeg")
        return f"data:{media_type};base64,{data}"
    return ""


def file_summary(block: dict) -> str:
    btype = block.get("type")
    filename = str(block.get("filename", "") or "")
    media_type = str(block.get("media_type", "application/octet-stream") or "application/octet-stream")
    if btype == "file_url":
        return f"[file_url: {filename or block.get('url', '')}; media_type={media_type}]"
    if btype == "file_base64":
        data = str(block.get("data", ""))
        return f"[file_base64: {filename}; media_type={media_type}; {len(data)} chars]"
    if btype == "file_id":
        return f"[file_id: {filename or block.get('file_id', '')}; media_type={media_type}]"
    return "[file]"


def image_summary(block: dict) -> str:
    btype = block.get("type")
    detail = block.get("detail", "auto")
    if btype == "image_url":
        url = str(block.get("url", ""))
        return f"[image_url: {url}; detail={detail}]"
    if btype == "image_base64":
        data = str(block.get("data", ""))
        media_type = str(block.get("media_type", "image/jpeg") or "image/jpeg")
        return f"[image_base64: {media_type}; {len(data)} chars; detail={detail}]"
    return "[image]"


def _extract_image_url(block: dict) -> tuple[str, str]:
    detail = str(block.get("detail", "auto") or "auto")
    raw = block.get("image_url")
    if isinstance(raw, dict):
        detail = str(raw.get("detail", detail) or detail)
        return str(raw.get("url", "") or ""), detail
    if isinstance(raw, str):
        return raw, detail
    return str(block.get("url", "") or ""), detail


def _file_url_block(block: dict) -> dict:
    return {
        "type": "file_url",
        "url": str(block.get("url", "") or block.get("file_url", "") or ""),
        "filename": str(block.get("filename", "") or ""),
        "media_type": str(block.get("media_type", "application/pdf") or "application/pdf"),
    }


def _file_base64_block(block: dict, *, data: str) -> dict:
    return {
        "type": "file_base64",
        "data": data,
        "filename": str(block.get("filename", "") or ""),
        "media_type": str(block.get("media_type", "application/pdf") or "application/pdf"),
    }


def _file_id_block(block: dict, *, file_id: str) -> dict:
    return {
        "type": "file_id",
        "file_id": file_id,
        "filename": str(block.get("filename", "") or ""),
        "media_type": str(block.get("media_type", "application/pdf") or "application/pdf"),
    }
