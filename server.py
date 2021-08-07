#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module docstring:

サーバーの起動とそのインターファイス

"""
# Standard Library
import os
import csv
from typing import Any, List, Tuple

from flask import Flask, flash, jsonify, request, wrappers, send_file, render_template
from werkzeug.utils import secure_filename

from main import inference

"""
設定
"""
# サーバーの URL の取得
_FLASK_URL = os.environ.get("FLASK_URL", "http://localhost:2626")
FILE_COUNT = 0
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(["wav"])
XLSX_MIMETYPE = "audio/wav"

# アプリ
app = Flask(__name__)


@app.route("/", methods=["GET"])
def get_home() -> Any:
    return render_template("index.html", _url=_FLASK_URL)


@app.route(
    "/predict",
    methods=[
        "POST",
    ],
)
def predict() -> Any:

    filebuf = request.files.get("wave")
    if filebuf is None:
        return jsonify(message="no file"), 400
    elif XLSX_MIMETYPE != filebuf.mimetype:
        return jsonify(message="is not wav"), 415
    print()

    # ファイルのチェック
    if filebuf and allwed_file(str(filebuf.filename)):
        # 危険な文字を削除（サニタイズ処理）
        filename = secure_filename(str(filebuf.filename))
        # ファイルの保存
        check_dirs(["./tmp/"])
        path_tmp = f"./tmp/{FILE_COUNT}_{filename}"
        filebuf.save(path_tmp)

        kana, romaji = inference(path_tmp)

        return render_template('result.html', kana = kana, romaji=romaji), 200

    else:
        return 'error', 500


@app.route("/test", methods=["GET", "POST"])
def test() -> Tuple[str, int]:
    return "試験成功です。", 200


def allwed_file(filename: str) -> int:
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def check_dirs(dirs: List[str]) -> None:
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 2626)))
