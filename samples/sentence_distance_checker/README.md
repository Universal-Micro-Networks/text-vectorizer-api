# sentence-distance-checker

## 前提

- juman++のv2をインストールしておく
  - brewではv1がインストールされてしまうので、ソースコードからインストールする
    - model付きの300Mほどあるやつをダウンロードする
    - https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
  - バイナリはパスを通しておくか、.venv/binの下に置くようにする

## 実行方法
```shell
export PATH=.venv/bin:$PATH && rye run uvicorn src.text_vectroizer_api.app:app --reload --port 9000
```
