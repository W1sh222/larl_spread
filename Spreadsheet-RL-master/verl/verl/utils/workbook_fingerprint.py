# Workbook content fingerprint (RewardFlow state identity 用).
# raw xlsx bytes 随序列化器 (openpyxl / Excel COM) 变化, 因此必须 parse 后对
# canonical content representation 做 hash; 文件路径 / timestamp / zip metadata 不参与.

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

STATUS_OK = "ok"
STATUS_UNKNOWN = "unknown"

DEFAULT_MAX_FILE_MB = 100.0  # 与 tool 侧 max_file_size_mb 对齐
DEFAULT_MAX_CELLS = 400_000
DEFAULT_MAX_DIMS = 200_000  # row/column dimension 记录上限 (Excel 常为每个 used column 写 dim)
_NO_CACHE = "<no_cache>"
_CACHE_MAX_ENTRIES = 512


def _norm_value(v: Any) -> str:
    if v is None:
        return "~"
    if isinstance(v, bool):
        return f"b:{v}"
    if isinstance(v, (int, float)):
        return f"n:{v!r}"
    if isinstance(v, str):
        return f"s:{v}"
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return f"d:{v.isoformat()}"
    # 未知类型只记录类型名, 避免不稳定 repr 进入 hash.
    return f"t:{type(v).__name__}"


def _formula_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    text = getattr(v, "text", None)
    if isinstance(text, str):
        return text
    return f"<{type(v).__name__}>"


def _border_sig(cell) -> str:
    # Border 无自定义 __repr__ (Serializable 不覆盖), str() 会带入内存地址 -> 非决定论.
    # 逐边提取 (style, color) 的确定性表示.
    b = cell.border
    if b is None:
        return ""
    parts = []
    for name in ("left", "right", "top", "bottom", "diagonal"):
        side = getattr(b, name, None)
        if side is None:
            parts.append(f"{name}:,")
        else:
            color = getattr(side, "color", None)
            color_rec = ""
            if color is not None:
                color_rec = str(getattr(color, "rgb", None) or getattr(color, "indexed", None) or getattr(color, "theme", None))
            parts.append(f"{name}:{getattr(side, 'style', None)},({color_rec})")
    parts.append(f"ds:{getattr(b, 'diagonalUp', None)},{getattr(b, 'diagonalDown', None)}")
    return ";".join(parts)


def _style_sig(cell) -> str:
    # 保守策略: 非空 cell 的 number_format/font/fill/alignment/border 一律纳入 fingerprint,
    # 避免 "仅格式不同" 被误合并为同一 state (format_range 是合法 action).
    # number_format: General 与 None 视为等价 (均原始显示), 避免无关序列化器差异碎片化.
    try:
        font = cell.font
        fill = cell.fill
        align = cell.alignment
        nf = cell.number_format
        # None/空 与 openpyxl getter 默认值 "General" 等价 (均原始显示, getter 从不返回 None);
        # "@" (Text) 是与 General 语义不同的显式格式, 必须区分, 不并入默认.
        nf_rec = "" if nf in (None, "", "General") else nf
        parts = (
            nf_rec,
            font.b, font.i, font.u, font.sz, font.name,
            getattr(font.color, "rgb", None) if font.color is not None else None,
            getattr(font.color, "theme", None) if font.color is not None else None,
            getattr(fill, "patternType", None),
            getattr(getattr(fill, "fgColor", None), "rgb", None),
            getattr(getattr(fill, "fgColor", None), "theme", None),
            align.horizontal, align.vertical, align.wrap_text,
            _border_sig(cell),
        )
        return ",".join(str(p) for p in parts)
    except Exception:
        return "style_unavailable"


def _unknown(reason: str) -> dict[str, Any]:
    return {"status": STATUS_UNKNOWN, "reason": str(reason), "fingerprint": None}


def _canonicalize_workbook(
    wb: Any,
    reload_workbook: Any,
    *,
    max_cells: int,
    max_dims: int,
    t0: float,
) -> dict[str, Any]:
    """已加载 workbook -> canonical content repr -> sha256.

    rollout 侧 (从路径加载) 与评估侧 (从提交的 bytes 加载) 共用本函数, 以保证
    canonicalization 规则完全一致: 同一内容 -> 同一 fingerprint.
    reload_workbook(data_only: bool) 用于重新打开同一来源以读 formula cache 值.
    """
    try:
        lines: list[str] = []
        cell_lines: list[tuple[int, int, int, str]] = []  # (sheet_idx, row, col, record)
        formula_coords: list[tuple[int, str]] = []  # (sheet_idx, coordinate)
        cell_count = 0

        for si, ws in enumerate(wb.worksheets):
            lines.append(f"S|{si}|{ws.title}|{ws.sheet_state}")
            try:
                merged = sorted(str(r) for r in ws.merged_cells.ranges)
            except Exception:
                merged = []
            if merged:
                lines.append(f"M|{si}|" + ";".join(merged))
            try:
                hidden_rows = sorted(int(k) for k, d in ws.row_dimensions.items() if getattr(d, "hidden", False))
                hidden_cols = sorted(str(k) for k, d in ws.column_dimensions.items() if getattr(d, "hidden", False))
            except Exception:
                hidden_rows, hidden_cols = [], []
            if hidden_rows:
                lines.append(f"HR|{si}|" + ",".join(str(r) for r in hidden_rows))
            if hidden_cols:
                lines.append(f"HC|{si}|" + ",".join(hidden_cols))

            # Row/column dimensions: hidden 之外, format_range 会直接写 height / width (可成为
            # 任务目标), 因此纳入稀疏 dim (仅已存在的 entry; .items() 不创建). 不纳入
            # dimension-level style index (跨文件不稳定) 与 outline, 见 README 已知限制.
            try:
                sf = ws.sheet_format
                lines.append(
                    f"SF|{si}|{getattr(sf, 'defaultRowHeight', None)}|"
                    f"{getattr(sf, 'defaultColWidth', None)}|{getattr(sf, 'customHeight', None)}"
                )
            except Exception:
                pass
            try:
                rd_items = sorted(((int(k), d) for k, d in ws.row_dimensions.items()), key=lambda t: t[0])
                cd_items = sorted(((str(k), d) for k, d in ws.column_dimensions.items()), key=lambda t: t[0])
            except Exception:
                rd_items, cd_items = [], []
            if len(rd_items) + len(cd_items) > max_dims:
                return _unknown("too_many_dims")
            for k, d in rd_items:
                lines.append(
                    f"RD|{si}|{k}|{getattr(d, 'hidden', None)}|"
                    f"{getattr(d, 'height', None)}|{getattr(d, 'customHeight', None)}"
                )
            for k, d in cd_items:
                lines.append(
                    f"CD|{si}|{k}|{getattr(d, 'hidden', None)}|"
                    f"{getattr(d, 'width', None)}|{getattr(d, 'bestFit', None)}|{getattr(d, 'customWidth', None)}"
                )

            for row in ws.iter_rows():
                for cell in row:
                    v = cell.value
                    if v is None:
                        # 空 cell 但带显式格式 (yellow fill / border 等) 也是环境 state.
                        # `_style is None` (从未分配 style) 是跳过的充分条件; 对 MergedCell
                        # (has_style 恒为 False 但 _style 可被分配) 同样成立.
                        # 已分配但恰为默认的会多几行记录, 不影响决定论与合并正确性.
                        if getattr(cell, "_style", None) is None:
                            continue
                        value_txt = "e:"
                    elif cell.data_type == "f":
                        value_txt = f"F:{_formula_text(v)}"
                        formula_coords.append((si, cell.coordinate))
                    else:
                        value_txt = _norm_value(v)
                    cell_count += 1
                    if cell_count > max_cells:
                        return _unknown("too_many_cells")
                    row_idx = int(cell.row) if cell.row is not None else -1
                    col_idx = int(cell.column) if cell.column is not None else -1
                    sig = _style_sig(cell)
                    cell_lines.append((si, row_idx, col_idx, f"C|{si}|{row_idx},{col_idx}|{value_txt}|{sig}"))

        try:
            for name in sorted(wb.defined_names.keys()):
                try:
                    lines.append(f"N|{name}|{wb.defined_names[name]}")
                except Exception:
                    lines.append(f"N|{name}|?")
        except Exception:
            pass

        for _, _, _, rec in sorted(cell_lines, key=lambda t: (t[0], t[1], t[2])):
            lines.append(rec)

        # formula cache value: 与 formula expression 分开记录 (显示值相同但公式不同 /
        # 公式相同但 cache 不同 都视为不同 state; 无法读取的 cache 记 <no_cache>, 不臆造).
        if formula_coords:
            try:
                wb_cache = reload_workbook(True)
                for si, coord in sorted(formula_coords):
                    try:
                        cv = wb_cache.worksheets[si][coord].value
                        coord_rec = _NO_CACHE if cv is None else _norm_value(cv)
                    except Exception:
                        coord_rec = _NO_CACHE
                    lines.append(f"X|{si}|{coord}|{coord_rec}")
                wb_cache.close()
            except Exception as exc:
                return _unknown(f"cache_parse_error:{type(exc).__name__}")

        digest = hashlib.sha256("\n".join(lines).encode("utf-8", errors="replace")).hexdigest()
        return {
            "status": STATUS_OK,
            "fingerprint": f"sha256:{digest}",
            "cells": cell_count,
            "parse_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
    except Exception as exc:  # 防御: 任何意外都归为 unknown, 不得污染 graph
        return _unknown(f"unexpected:{type(exc).__name__}")


def fingerprint_workbook(
    path: str,
    *,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_dims: int = DEFAULT_MAX_DIMS,
) -> dict[str, Any]:
    """返回 {"status": ok/unknown, "fingerprint": "sha256:<hex>"|None, "reason", "cells", "parse_ms"}."""
    p = Path(path)
    try:
        st = os.stat(p)
    except OSError:
        return _unknown("missing_file")
    if not os.path.isfile(p):
        return _unknown("not_a_file")
    if st.st_size > max_file_mb * 1024 * 1024:
        return _unknown("file_too_large")

    try:
        import openpyxl
    except ImportError:
        return _unknown("openpyxl_unavailable")

    def reload(data_only: bool):
        return openpyxl.load_workbook(filename=str(p), data_only=data_only, keep_links=False)

    t0 = time.perf_counter()
    try:
        wb = reload(False)
    except Exception as exc:  # 损坏文件 / 不支持的 feature
        return _unknown(f"parse_error:{type(exc).__name__}")

    try:
        return _canonicalize_workbook(wb, reload, max_cells=max_cells, max_dims=max_dims, t0=t0)
    finally:
        try:
            wb.close()
        except Exception:
            pass


def fingerprint_workbook_bytes(
    data: bytes,
    *,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_dims: int = DEFAULT_MAX_DIMS,
) -> dict[str, Any]:
    """从内存中的 workbook bytes 计算 canonical fingerprint (评估侧用).

    与 fingerprint_workbook(path) 共用 _canonicalize_workbook -> 同一内容同一指纹.
    不重新打开原始路径; 不直接 hash raw zip bytes (zip metadata 非决定论).
    计算失败/超限 -> status=unknown, fingerprint=None (不伪造成正常指纹).
    """
    if data is None:
        return _unknown("no_bytes")
    try:
        n = len(data)
    except TypeError:
        return _unknown("invalid_bytes")
    if n <= 0:
        return _unknown("empty_bytes")
    if n > max_file_mb * 1024 * 1024:
        return _unknown("file_too_large")

    try:
        import io

        import openpyxl
    except ImportError:
        return _unknown("openpyxl_unavailable")

    def reload(data_only: bool):
        return openpyxl.load_workbook(io.BytesIO(data), data_only=data_only, keep_links=False)

    t0 = time.perf_counter()
    try:
        wb = reload(False)
    except Exception as exc:
        return _unknown(f"parse_error:{type(exc).__name__}")

    try:
        return _canonicalize_workbook(wb, reload, max_cells=max_cells, max_dims=max_dims, t0=t0)
    finally:
        try:
            wb.close()
        except Exception:
            pass


def stat_signature(path: str) -> Optional[tuple]:
    """(dev, ino, size, mtime_ns) — 仅作 cache 失效判断用, 不是 state identity."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def fingerprint_workbook_cached(path: str, cache: "OrderedDict[tuple, dict]") -> dict[str, Any]:
    """带文件签名缓存的 fingerprint. 仅缓存 ok 结果 (unknown 可能因后续 writeback 转好)."""
    sig = stat_signature(path)
    if sig is None:
        return _unknown("missing_file")
    hit = cache.get(sig)
    if hit is not None:
        cache.move_to_end(sig)
        return {**hit, "cache_hit": True}
    result = fingerprint_workbook(path)
    if result.get("status") == STATUS_OK:
        cache[sig] = result
        cache.move_to_end(sig)
        while len(cache) > _CACHE_MAX_ENTRIES:
            cache.popitem(last=False)
    return {**result, "cache_hit": False}
