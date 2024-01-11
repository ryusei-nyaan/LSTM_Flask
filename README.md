# LSTM_Flask
LSTMによる時系列予測を行う簡易WebアプリをFlaskで実装

# 使い方
１．Flask_LSTMフォルダをダウンロード

２．dockerを起動してpowershell (macだとターミナル)でディレクトリをFlask_LSTM直下に移動します

３．docker compose upを打ち込みます（必要に応じてデタッチモードでお願いします）
```shell
docker compose up
```

４．結構待つ

５．起動したと思われたら，任意のブラウザでlocalhost:5000にアクセス．

※ポート5000が使われていると表示された場合，yamlファイルのports:- "5000:80"の5000を5001などにしてから2に戻ってください

６．過去何点分のデータを使って予測するか，未来何点分を予測するか決めてください（7のtest.csvを使う場合，過去時間は32,未来120点がおすすめです）

７．csvファイルはFlask_LSTM/flasktestLSTM/test_data配下のtest.csvを使ってください．正弦波のデータです（持ち前のcsv時系列データでも良いですが現時点で動作は保証できないです）

８．結構待つ

９．結果が表示されます．（この後の遷移を想定して作成していないため，閉じてしまってください）

１０．遊び終わったらコンテナを止めてください．容量を最適化できていないため，削除してください．
