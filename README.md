# TiDB-Bedrock
TiDB ServerlessとAmazon Bedrockを使用したRAGアプリケーション  
Zenn記事へのリンク→[TiDB (Serverless) × Amazon Bedrockで始めるRAGアプリケーション入門！](https://zenn.dev/koiping/articles/a4362c8b1c0ee8)

## ローカル環境で検証する場合は、下記の手順で実行してください。
### 必要なもの
- ベクトル検索機能が有効化されているTiDB Serverlessクラスター
- Python 3.8 もしくはそれ以上のバージョン
- AWSのアカウントを持っており、Amazon Bedrockが利用できる状態(今回利用するモデルが有効化されていること)

## 実行方法
### このGitHubのリポジトリをローカル環境にCloneしてください。

```bash
git clone https://github.com/Yoshiitaka/TiDB-Bedrock.git
```

### venvの環境を用意し、Active化します。

```bash
python -m venv .venv
source .venv/bin/activate
```

### 依存関係のあるライブラリ、パッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 環境変数にTiDBへの接続情報を記載します。
サンプルのファイル「sample-env」を用意しています。
このファイルを参考に「.env」ファイルを作成します。
下記の接続情報をベクトル検索を有効化したTiDB Serverlessクラスターの接続情報で更新します。

```.env
TIDB_HOSTNAME='xxxxxxx'
TIDB_USERNAME='xxxxxx.root'
TIDB_PASSWORD='xxxxxx'
TIDB_DATABASE_NAME='test'
```

### ローカル環境のAWS Credentials情報が今回利用するAmazon Bedrockのモデルが有効になっているアカウント、リージョン、利用権限が振られていることを確認します。

```
cat ~/.aws/credentials
```

## ローカルでのサーバ起動方法
### 事前にTiDBのベクトルカラムにパブリックに公開されているTiDBのブログをEmbeddingしたベクトルデータを流し込みます。

```bash
python prepare.py
```

### ローカルでUI画面を起動し、実際のRAGを体験する

```bash
# runserver
streamlit run main.py
```

立ち上がるローカルのポート番号は8501です。
なので、下記へブラウザよりアクセスしてみてください。

[http://localhost:8501/](http://localhost:8501/)
