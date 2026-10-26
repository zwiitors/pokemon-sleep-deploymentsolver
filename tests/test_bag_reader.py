"""bag_reader の回帰テスト。実行: python -m pytest tests"""

import json
import os
import sys

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bag_reader  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def load_fixture(name):
    image = cv2.imread(os.path.join(FIXTURES, f"{name}.png"))
    with open(os.path.join(FIXTURES, f"{name}.json"), encoding="utf-8") as f:
        expected = {c["ingredient"]: c["count"] for c in json.load(f)["cells"]}
    return image, expected


def degrade(image, width, quality):
    """別端末の解像度・JPEG圧縮を模擬する。"""
    h, w = image.shape[:2]
    resized = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_reads_fixture_exactly():
    image, expected = load_fixture("bag_top")
    result = bag_reader.read_bag([image])
    assert result.counts == expected
    assert not result.unknown_cells
    assert not result.low_confidence
    assert not result.order_warnings
    # カカオと大豆の間にあるシッポは表示されていない＝0
    assert result.zero == ["シッポ"]
    # コーヒーより後ろは画面外の可能性があるので未確認
    assert result.unconfirmed == ["カボチャ", "アボカド"]


@pytest.mark.parametrize("width", [720, 828, 1170, 1440])
@pytest.mark.parametrize("quality", [50, 90])
def test_other_resolutions_and_jpeg(width, quality):
    image, expected = load_fixture("bag_top")
    result = bag_reader.read_bag([degrade(image, width, quality)])
    assert result.counts == expected
    assert not result.unknown_cells
    assert not result.low_confidence


def test_scrolled_screenshots_are_merged():
    image, expected = load_fixture("bag_top")
    # 上半分と、少しスクロールした下側の2枚に分けて読ませる（重複あり）
    top = image[:1600]
    bottom = image[700:]
    result = bag_reader.read_bag([top, bottom])
    assert result.counts == expected
    assert not result.conflicts


def test_missing_rows_are_unconfirmed_when_list_top_not_visible():
    image, _ = load_fixture("bag_top")
    # 先頭行が見えない画像だけなら、ネギ〜肉は 0 ではなく未確認
    result = bag_reader.read_bag([image[900:]])
    assert "ネギ" in result.unconfirmed
    assert "ネギ" not in result.zero


def test_unknown_icon_is_not_misidentified(tmp_path):
    """見本のない食材（新食材を想定）は別の食材と誤判定せず「不明」になる。"""
    image, expected = load_fixture("bag_top")
    for removed in expected:
        icon_dir = tmp_path / removed
        icon_dir.mkdir()
        for name in os.listdir(bag_reader.ICON_DIR):
            if name.split("__")[0].removesuffix(".png") != removed:
                (icon_dir / name).write_bytes(
                    open(os.path.join(bag_reader.ICON_DIR, name), "rb").read()
                )
        templates = bag_reader.Templates(icon_dir=str(icon_dir))
        result = bag_reader.read_bag([image], templates)
        assert removed not in result.counts
        assert len(result.unknown_cells) == 1, removed
        assert result.unknown_cells[0].count == expected[removed]
        others = {k: v for k, v in expected.items() if k != removed}
        assert result.counts == others
