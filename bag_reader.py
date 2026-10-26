"""ポケモンスリープの「バッグ > 食材」画面のスクリーンショットから食材数を読み取る。

汎用OCRは使わず、OpenCV のテンプレート照合だけで読み取る。
- 食材の判定: アイコン（クリーム色の円）を assets/ingredient_icons/*.png と照合
- 個数の読み取り: バッジ「xNN」の数字を assets/digits/*.png と照合

画面は横幅を基準に拡大縮小されるため、入力は横幅 1080px に正規化してから処理する。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets", "ingredient_icons")
DIGIT_DIR = os.path.join(BASE_DIR, "assets", "digits")

NORM_WIDTH = 1080
CIRCLE_SIZE = 192  # 横幅1080pxでのアイコン円の直径
ICON_SIZE = 64  # 照合用に縮小するアイコンの大きさ
DIGIT_SIZE = (24, 32)  # 照合用に正規化する数字の大きさ (幅, 高さ)
LIST_TOP_Y = (560, 720)  # リスト先頭行の円の上端がこの範囲なら「リストの先頭が見えている」

ICON_MATCH_THRESHOLD = 0.74  # これ未満のアイコンは「不明」（正解は0.82以上、別食材は0.65以下だった）
ICON_MATCH_MARGIN = 0.10  # 1位と2位の差がこれ未満なら「不明」
DIGIT_MATCH_THRESHOLD = 0.85  # これ未満の数字があれば確からしさを下げる（正解は0.93以上、別数字は0.72以下だった）

# ゲーム内の「表示順 - デフォルト」の並び（個数0の食材は表示されない）
DEFAULT_ORDER = [
    "ネギ",
    "キノコ",
    "卵",
    "イモ",
    "リンゴ",
    "ハーブ",
    "肉",
    "牛乳",
    "ミツ",
    "オイル",
    "ジンジャー",
    "トマト",
    "カカオ",
    "シッポ",
    "大豆",
    "コーン",
    "コーヒー",
    "カボチャ",
    "アボカド",
]

# ゲーム内の食材名 → ソルバーの食材名
GAME_NAMES = {
    "ふといながねぎ": "ネギ",
    "あじわいキノコ": "キノコ",
    "とくせんエッグ": "卵",
    "ほっこりポテト": "イモ",
    "とくせんリンゴ": "リンゴ",
    "げきからハーブ": "ハーブ",
    "マメミート": "肉",
    "モーモーミルク": "牛乳",
    "あまいミツ": "ミツ",
    "ピュアなオイル": "オイル",
    "あったかジンジャー": "ジンジャー",
    "あんみんトマト": "トマト",
    "リラックスカカオ": "カカオ",
    "ワカクサ大豆": "大豆",
    "ワカクサコーン": "コーン",
    "めざましコーヒー": "コーヒー",
}


@dataclass
class Cell:
    """スクリーンショット内の1マス（食材1種類）の読み取り結果。"""

    ingredient: str | None  # 判定できなければ None
    count: int | None  # 読めなければ None
    icon_score: float
    digit_score: float
    icon_image: np.ndarray = field(repr=False)  # 確認表示用（RGB）
    position: tuple[int, int] = (0, 0)  # 正規化後の円の左上 (x, y)

    @property
    def confident(self) -> bool:
        return (
            self.ingredient is not None
            and self.count is not None
            and self.icon_score >= ICON_MATCH_THRESHOLD
            and self.digit_score >= DIGIT_MATCH_THRESHOLD
        )


@dataclass
class ScreenResult:
    cells: list[Cell]
    shows_list_top: bool  # リストの先頭行が見えているか


@dataclass
class BagResult:
    counts: dict[str, int]  # 読み取れた食材数
    zero: list[str]  # 表示されていない＝0と判断できた食材
    unconfirmed: list[str]  # 画面外の可能性があり判断できない食材
    unknown_cells: list[Cell]  # アイコンを判定できなかったマス
    low_confidence: list[Cell]  # 判定はできたが確からしさが低いマス
    conflicts: dict[str, list[int]]  # 複数枚で値が食い違った食材
    order_warnings: list[str]  # 並び順と矛盾する箇所


# ---------------------------------------------------------------------------
# 画像の前処理・検出
# ---------------------------------------------------------------------------


def load_image(data: bytes | str | np.ndarray) -> np.ndarray:
    """bytes / ファイルパス / BGR配列 を受け取り、横幅1080pxのBGR画像を返す。"""
    if isinstance(data, np.ndarray):
        img = data
    elif isinstance(data, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像を読み込めませんでした。")
    h, w = img.shape[:2]
    if w != NORM_WIDTH:
        img = cv2.resize(
            img,
            (NORM_WIDTH, round(h * NORM_WIDTH / w)),
            interpolation=cv2.INTER_AREA if w > NORM_WIDTH else cv2.INTER_CUBIC,
        )
    return img


def find_circles(img: np.ndarray) -> list[tuple[int, int]]:
    """クリーム色のアイコン円を探し、欠けていない円の左上座標を返す（上→下、左→右）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 15, 240]), np.array([35, 60, 255]))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    circles = []
    for x, y, w, h, area in stats[1:]:
        # 円の外形がほぼ CIRCLE_SIZE 四方のものだけ採用（画面端で切れた円や卵の中身は除外）
        if abs(w - CIRCLE_SIZE) <= 12 and abs(h - CIRCLE_SIZE) <= 12 and area > 8000:
            # 下に続くバッジ・数字が画面内に収まっているか
            if y + CIRCLE_SIZE + 10 <= img.shape[0]:
                circles.append((int(x), int(y)))
    circles.sort(key=lambda p: (round(p[1] / 50), p[0]))
    return circles


def icon_crop(img: np.ndarray, x: int, y: int) -> np.ndarray:
    return img[y : y + CIRCLE_SIZE, x : x + CIRCLE_SIZE]


def icon_feature(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """照合用のアイコン特徴（縮小画像, 前景マスク）。

    クリーム色・白の背景はどのアイコンでも同じで識別の邪魔になるため、
    前景（食材の絵）の部分だけを比べられるようにマスクを作る。
    バッジが重なる右下部分は比較から除く。
    """
    small = cv2.resize(crop, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    # 背景判定は元の解像度で行い、絵の輪郭の内側を塗りつぶしてシルエットにする
    # （牛乳瓶や卵のように白っぽい部分があっても形が安定する）
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
    background = (hsv[..., 1] < 70) & (hsv[..., 2] > 225)
    fg = (~background).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 150]  # 細かなノイズを除く
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    mask = cv2.resize(filled, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA) > 127
    cut = int(ICON_SIZE * 0.72)
    mask[cut:, int(ICON_SIZE * 0.45) :] = False
    return small.astype(np.float32), mask


def icon_score(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """前景マスクの和集合の範囲で正規化相互相関をとる。形の重なり具合も掛け合わせる。"""
    (ia, ma), (ib, mb) = a, b
    union = ma | mb
    if not union.any():
        return 0.0
    overlap = (ma & mb).sum() / union.sum()  # 形（シルエット）の一致度
    return _ncc(ia[union], ib[union]) * overlap


def badge_crop(img: np.ndarray, x: int, y: int) -> np.ndarray:
    """バッジ「xNN」の文字部分を切り出す。"""
    return img[y + 148 : y + 200, x + 95 : x + 230]


def split_glyphs(badge: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """バッジの文字を1文字ずつ切り出す（左→右）。先頭の「x」も含む。

    文字の位置は二値化で求め、切り出す画像は濃淡（白黒反転したグレー）のまま返す。
    二値化の境目の揺れに左右されにくくするため。
    """
    gray = cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = [
        (x, y, w, h)
        for x, y, w, h, area in stats[1:]
        if h >= 12 and area >= 20  # 枠線の影などの小さなノイズを除く
    ]
    boxes.sort(key=lambda b: b[0])
    if len(boxes) >= 2:
        # 先頭は「x」。数字は背が高くそろっているので、低いノイズを除く
        digit_h = max(b[3] for b in boxes[1:])
        boxes = boxes[:1] + [b for b in boxes[1:] if b[3] >= digit_h * 0.8]
    inverted = 255 - gray
    glyphs = []
    for x, y, w, h in boxes:
        x0, y0 = max(0, x - 1), max(0, y - 1)
        g = inverted[y0 : y + h + 1, x0 : x + w + 1]
        glyphs.append((x, y, w, h, g))
    return glyphs


def glyph_feature(g: np.ndarray) -> np.ndarray:
    """縦横比を保ったまま高さをそろえ、中央寄せで固定サイズにする（「1」が潰れないように）。"""
    tw, th = DIGIT_SIZE
    h, w = g.shape[:2]
    nw = max(1, min(tw, round(w * th / h)))
    resized = cv2.resize(g, (nw, th), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), np.uint8)
    off = (tw - nw) // 2
    canvas[:, off : off + nw] = resized
    return canvas.astype(np.float32)


def digit_score(a: np.ndarray, b: np.ndarray, shift: int = 2) -> float:
    """数字同士の類似度。切り出し位置の数pxのずれを許して最大の相関をとる。"""
    best = -1.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            best = max(best, _ncc(np.roll(a, (dy, dx), axis=(0, 1)), b))
    return best


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """正規化相互相関（-1〜1）。"""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# 見本の読み込みと照合
# ---------------------------------------------------------------------------


class Templates:
    def __init__(self, icon_dir: str = ICON_DIR, digit_dir: str = DIGIT_DIR):
        self.icons = self._load_dir(icon_dir, cv2.IMREAD_COLOR, icon_feature)
        self.digits = self._load_dir(digit_dir, cv2.IMREAD_GRAYSCALE, glyph_feature)

    @staticmethod
    def _load_dir(path, flag, feature):
        result = {}
        if not os.path.isdir(path):
            return result
        for name in sorted(os.listdir(path)):
            if not name.endswith(".png"):
                continue
            # 日本語ファイル名でも読めるよう imdecode を使う
            data = np.fromfile(os.path.join(path, name), np.uint8)
            img = cv2.imdecode(data, flag)
            label = os.path.splitext(name)[0].split("__")[0]  # 「ネギ__2.png」も可
            result.setdefault(label, []).append(feature(img))
        return result

    def match_icon(self, crop: np.ndarray) -> tuple[str | None, float]:
        feat = icon_feature(crop)
        scores = sorted(
            ((max(icon_score(feat, f) for f in feats), label) for label, feats in self.icons.items()),
            reverse=True,
        )
        if not scores:
            return None, 0.0
        best_score, best = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if best_score < ICON_MATCH_THRESHOLD or best_score - second < ICON_MATCH_MARGIN:
            return None, best_score
        return best, best_score

    def read_count(self, badge: np.ndarray) -> tuple[int | None, float]:
        glyphs = split_glyphs(badge)
        if len(glyphs) < 2:
            return None, 0.0
        digits, scores = [], []
        for *_, g in glyphs[1:]:  # 先頭は「x」
            feat = glyph_feature(g)
            best, best_score = None, -1.0
            for label, feats in self.digits.items():
                score = max(digit_score(feat, f) for f in feats)
                if score > best_score:
                    best, best_score = label, score
            digits.append(best)
            scores.append(best_score)
        if any(d is None for d in digits):
            return None, 0.0
        return int("".join(digits)), min(scores)


_default_templates: Templates | None = None


def default_templates() -> Templates:
    global _default_templates
    if _default_templates is None:
        _default_templates = Templates()
    return _default_templates


# ---------------------------------------------------------------------------
# 読み取り本体
# ---------------------------------------------------------------------------


def read_screen(image, templates: Templates | None = None) -> ScreenResult:
    templates = templates or default_templates()
    img = load_image(image)
    circles = find_circles(img)
    cells = []
    for x, y in circles:
        crop = icon_crop(img, x, y)
        ingredient, icon_score = templates.match_icon(crop)
        count, digit_score = templates.read_count(badge_crop(img, x, y))
        cells.append(
            Cell(
                ingredient=ingredient,
                count=count,
                icon_score=icon_score,
                digit_score=digit_score,
                icon_image=cv2.cvtColor(
                    img[y : y + CIRCLE_SIZE + 10, x : x + CIRCLE_SIZE + 40],
                    cv2.COLOR_BGR2RGB,
                ),
                position=(x, y),
            )
        )
    shows_top = bool(circles) and LIST_TOP_Y[0] <= circles[0][1] <= LIST_TOP_Y[1]
    return ScreenResult(cells=cells, shows_list_top=shows_top)


def read_bag(images, templates: Templates | None = None) -> BagResult:
    """複数枚のスクリーンショットを読み取り、食材ごとの個数にまとめる。

    個数0の食材はゲーム画面に表示されないため、各スクリーンショットで
    「見えている範囲（並び順で最初〜最後の食材の間）」にあるのに表示されて
    いない食材は 0 と判断する。範囲外の食材は画面外の可能性があるため未確認とする。
    """
    templates = templates or default_templates()
    order_index = {name: i for i, name in enumerate(DEFAULT_ORDER)}

    values: dict[str, list[int]] = {}
    covered: set[int] = set()
    unknown, low_conf, order_warnings = [], [], []

    for n, image in enumerate(images, start=1):
        screen = read_screen(image, templates)
        known = [c for c in screen.cells if c.ingredient is not None]
        for c in screen.cells:
            if c.ingredient is None:
                unknown.append(c)
                continue
            if not c.confident:
                low_conf.append(c)
            if c.count is not None:
                values.setdefault(c.ingredient, []).append(c.count)

        indices = [order_index[c.ingredient] for c in known if c.ingredient in order_index]
        for a, b in zip(indices, indices[1:]):
            if a >= b:
                order_warnings.append(
                    f"{n}枚目: {DEFAULT_ORDER[a]} → {DEFAULT_ORDER[b]} の並びが表示順と合いません"
                    "（アイコンの誤判定の可能性があります）"
                )
        # 不明なマスがあると、そのマスが範囲内の食材の可能性があるため範囲判断に使わない
        if indices and not any(c.ingredient is None for c in screen.cells):
            start = 0 if screen.shows_list_top else min(indices)
            covered.update(range(start, max(indices) + 1))

    counts, conflicts = {}, {}
    for name, vs in values.items():
        if len(set(vs)) > 1:
            conflicts[name] = vs
        counts[name] = max(set(vs), key=vs.count)

    zero = [
        name
        for i, name in enumerate(DEFAULT_ORDER)
        if name not in counts and i in covered
    ]
    unconfirmed = [
        name for name in DEFAULT_ORDER if name not in counts and name not in zero
    ]
    return BagResult(
        counts=counts,
        zero=zero,
        unconfirmed=unconfirmed,
        unknown_cells=unknown,
        low_confidence=low_conf,
        conflicts=conflicts,
        order_warnings=order_warnings,
    )


# ---------------------------------------------------------------------------
# 見本の作成（開発用）
# ---------------------------------------------------------------------------


def make_templates(image_path: str, labels: list[dict], icon_dir=ICON_DIR, digit_dir=DIGIT_DIR):
    """正解ラベル付きのスクリーンショットから見本画像を切り出して保存する。

    labels: 画面上の並び順（上→下、左→右）に {"ingredient": 名前, "count": 個数}
    既にある見本は上書きしない（同じ食材の別見本は「名前__2.png」として追加）。
    """
    os.makedirs(icon_dir, exist_ok=True)
    os.makedirs(digit_dir, exist_ok=True)
    img = load_image(image_path)
    circles = find_circles(img)
    if len(circles) != len(labels):
        raise ValueError(f"検出したマス数 {len(circles)} とラベル数 {len(labels)} が一致しません")

    def save(dirname, label, image):
        path = os.path.join(dirname, f"{label}.png")
        k = 2
        while os.path.exists(path):
            path = os.path.join(dirname, f"{label}__{k}.png")
            k += 1
        ok, buf = cv2.imencode(".png", image)
        buf.tofile(path)

    saved_icons, saved_digits = [], set()
    existing_digits = {
        os.path.splitext(f)[0].split("__")[0] for f in os.listdir(digit_dir)
    }
    existing_icons = {
        os.path.splitext(f)[0].split("__")[0] for f in os.listdir(icon_dir)
    }
    for (x, y), label in zip(circles, labels):
        if label["ingredient"] not in existing_icons:
            save(icon_dir, label["ingredient"], icon_crop(img, x, y))
            existing_icons.add(label["ingredient"])
            saved_icons.append(label["ingredient"])
        glyphs = split_glyphs(badge_crop(img, x, y))[1:]
        text = str(label["count"])
        if len(glyphs) != len(text):
            raise ValueError(f"{label['ingredient']}: 数字の切り出し数が {text} と一致しません")
        for ch, (*_, g) in zip(text, glyphs):
            if ch not in existing_digits:
                save(digit_dir, ch, g)
                existing_digits.add(ch)
                saved_digits.add(ch)
    return saved_icons, sorted(saved_digits)


if __name__ == "__main__":
    # 使い方: python bag_reader.py make <画像> <ラベルJSON>
    #         python bag_reader.py read <画像>...
    import sys

    if len(sys.argv) >= 4 and sys.argv[1] == "make":
        with open(sys.argv[3], encoding="utf-8") as f:
            labels = json.load(f)["cells"]
        icons, digits = make_templates(sys.argv[2], labels)
        print("追加したアイコン:", icons)
        print("追加した数字:", digits)
    elif len(sys.argv) >= 3 and sys.argv[1] == "read":
        r = read_bag(sys.argv[2:])
        print("読み取り:", r.counts)
        print("0と判断:", r.zero)
        print("未確認:", r.unconfirmed)
        print("不明なマス:", len(r.unknown_cells))
        for w in r.order_warnings:
            print("警告:", w)
    else:
        print(__doc__)
