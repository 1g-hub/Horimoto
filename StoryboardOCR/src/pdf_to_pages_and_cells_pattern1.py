# pdf_to_pages_and_cells.py

import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

# ======================
# CONFIG (ここだけ触ればOK)
# ======================
EPISODE = "free2" # "episode01", "trigger", "free1"
BASE_DIR = Path("data/free") / EPISODE  # Path("data"), Path("data/free")

PDF_PATH = BASE_DIR / f"{EPISODE}.pdf"

# 出力ディレクトリ
DIRS = {
    "pages": BASE_DIR / "pages",
    "cut": BASE_DIR / "cut",
    "picture": BASE_DIR / "picture",
    "action_memo": BASE_DIR / "action_memo",
    "dialogue": BASE_DIR / "dialogue",
    "time": BASE_DIR / "time",
}

# ページPNG化の解像度
PAGE_DPI = 1200

# 表セル切り抜き設定（座標系はPDF座標）
# 参考コードの値をそのまま移植

# free [9, 50, 230, 366, 500, 527]  
# free n [21, 62, 250, 390, 528, 556], Y0 = 173, CELL_H = 106
# trigger [49, 75, 208, 404, 534, 558], Y0 = 92, CELL_H = 112
# episode1 [23, 53, 290, 421, 543, 573]

X_COORDS = [21, 62, 250, 390, 528, 556]  # 列境界 (x0..x5) = 5列分 
Y0 = 173                                  # ヘッダ直下の開始y
CELL_H = 106
DATA_ROWS = 5

# 切り抜き解像度スケール（Matrix(2,2) なら 2倍）
CELL_SCALE = 2

# 最初のページを飛ばす（ヘッダ等想定）
SKIP_FIRST_PAGES_FOR_CELLS = 0  # 1なら2ページ目から切り抜き
# ======================


def ensure_dirs(dirs: dict[str, Path]) -> None:
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)


def save_pages_as_png(doc: fitz.Document, pages_dir: Path, dpi: int) -> None:
    # dpi -> scale: 72dpi が基準なので dpi/72
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    for page_index in tqdm(range(len(doc)), desc=f"{EPISODE} | PDF→pages PNG", unit="page"):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out = pages_dir / f"page{page_index + 1}.png"
        pix.save(out)


def crop_table_cells(
    doc: fitz.Document,
    out_dirs: dict[str, Path],
    x_coords: list[int],
    y0: int,
    cell_h: int,
    data_rows: int,
    scale: int,
    skip_first_pages: int,
) -> None:
    # 列の対応（参考コード column_names を、あなたのディレクトリ構成に合わせて変換）
    col_keys = ["cut", "picture", "action_memo", "dialogue", "time"]

    if len(x_coords) != len(col_keys) + 1:
        raise ValueError(f"X_COORDS は {len(col_keys)+1} 個必要です（今は {len(x_coords)} 個）")

    mat = fitz.Matrix(scale, scale)

    page_indices = list(range(skip_first_pages, len(doc)))
    for page_index in tqdm(page_indices, desc=f"{EPISODE} | cells crop", unit="page"):
        page = doc[page_index]
        page_no = page_index + 1  # ファイル名は 1-based

        for row in range(1, data_rows + 1):
            y_top = y0 + (row - 1) * cell_h
            y_bottom = y0 + row * cell_h

            for col, key in enumerate(col_keys):
                x0 = x_coords[col]
                x1 = x_coords[col + 1]
                rect = fitz.Rect(x0, y_top, x1, y_bottom)

                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                out = out_dirs[key] / f"page{page_no}_row{row}.png"
                pix.save(out)


def main() -> None:
    ensure_dirs(DIRS)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF が見つかりません: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    try:
        # 1) ページ全体を pages/ に保存
        save_pages_as_png(doc, DIRS["pages"], PAGE_DPI)

        # 2) 表のセルを各ディレクトリに切り抜き保存
        crop_table_cells(
            doc=doc,
            out_dirs=DIRS,
            x_coords=X_COORDS,
            y0=Y0,
            cell_h=CELL_H,
            data_rows=DATA_ROWS,
            scale=CELL_SCALE,
            skip_first_pages=SKIP_FIRST_PAGES_FOR_CELLS,
        )
    finally:
        doc.close()

    print("Finished.")


if __name__ == "__main__":
    main()
