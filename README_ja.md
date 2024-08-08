# Shin Rakuda 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 目次
- [Shin Rakuda](#shin-rakuda)
  - [目次](#目次)
  - [概要](#概要)
  - [主な機能](#主な機能)
  - [前提条件](#前提条件)
  - [設定](#設定)
    - [モデル](#モデル)
    - [評価データセット](#評価データセット)
    - [判断モデル](#判断モデル)
    - [評価設定](#評価設定)
  - [インストール](#インストール)
    - [代替の依存関係管理](#代替の依存関係管理)
  - [使用方法](#使用方法)
  - [今後の課題](#今後の課題)
  - [貢献方法](#貢献方法)
  - [トラブルシューティング](#トラブルシューティング)
  - [ライセンス](#ライセンス)
  - [引用](#引用)
  - [参考文献](#参考文献)

## 概要

Shin Rakudaは、様々な言語モデル（LLM）の性能を指定されたデータセットでベンチマークするための強力かつ柔軟なツールです。研究者や開発者に、データセットの読み込み、モデルの選択、ベンチマークプロセスの実行、結果の可視化を行うための使いやすいインターフェースを提供します。

## 主な機能

- 複数の推論ライブラリ（HuggingfaceとVLLM）のサポート
- モデル、データセット、評価パラメータの柔軟な設定
- 使いやすいコマンドラインインターフェース
- ベンチマーク結果の可視化
- APIベースのモデルとローカルモデルの両方をサポート

## 前提条件

- Python 3.9以上
- 依存関係管理用のpipまたはPoetry
- 必要なモデルAPIへのアクセス（APIベースのモデルを使用する場合）
- ローカルモデルを実行するための十分な計算リソース（該当する場合）

## 設定

1. `.env.example`ファイルを`.env`にコピーし、必要に応じてモデルのAPIキーを設定します：

   ```bash
   cp .env.example .env
   ```

2. `config.yaml`ファイルを編集してプロジェクトを設定します。設定ファイルは以下のセクションに分かれています：

   - Models: ベンチマークするLLMを定義
   - Evaluation Datasets: 評価用のデータセットを指定
   - Judge Model: 応答を判断するモデルを設定
   - Evaluation Configurations: ディレクトリやその他の評価パラメータを設定

   各設定オプションの詳細な説明については、`config_template.yaml`ファイルのコメントを参照してください。

### モデル

```yaml
# APIモデル
models:
  - model_name: string
    api: boolean # モデルがAPI経由で推論を行うかどうか、APIモデルの場合はデフォルトでTrue
    provider: string # モデルのプロバイダ
# ローカルモデル
  - model_name: string # 任意の名前を指定可能
    api: boolean # モデルがAPI経由で推論を行うかどうか、ローカルモデルの場合はデフォルトでFalse
    provider: string # モデルのホスティングプロバイダ、デフォルトはhuggingface
    system_prompt: string 
    do_sample: boolean
    vllm_config: # vllm設定セクション
      model: string # モデルのフルネームまたはモデルID
      max_model_len: int # 最大モデル長
    vllm_sampling_params: # vllmサンプリングパラメータセクション
      temperature: float # 温度
      top_p: float # top p
      max_tokens: int # 最大トークン数
      repetition_penalty: float # 繰り返しペナルティ
    hf_pipeline: # huggingfaceパイプラインセクション
      task: string
      model: string # モデルのフルネームまたはモデルID
      torch_dtype: string 
      max_new_tokens: int
      device_map: string
      trust_remote_code: boolean
      return_full_text: boolean
    hf_chat_template: # huggingfaceチャットテンプレートセクション
      chat_template: string # 完全なチャットテンプレートまたは`ChatML`などのチャットテンプレートフォーマット
      tokenize: boolean # チャットテンプレートをトークン化するかどうか
      add_generation_prompt: boolean # 生成プロンプトを追加するかどうか
```

参考文献：

- [VLLM Engine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine)
- [VLLM Sampling Parameters](https://docs.vllm.ai/en/latest/dev/sampling_params.html)
- [Huggingface Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline)
- [Huggingface Chat Template](https://huggingface.co/docs/transformers/en/chat_templating)

HFまたはVLLMの設定パラメータを必要に応じて追加してください。Rakudaはそれに応じて処理を行います。どちらの推論ライブラリでもサポートされていないパラメータがある場合、Rakudaは正常に動作しません。

### 評価データセット

```yaml
eval_datasets:
  - dataset_name: string # データセット名
    judge_prompt_template: string # 判断プロンプトテンプレート
    num_questions: int # オプション、評価する質問の数
    random_questions: boolean # オプション、num_questionsが指定されている場合に質問をランダムに選択するかどうか
    use_jinja: boolean # 判断プロンプトにJinjaテンプレートを使用するかどうか
    score_keyword: string # モデル出力からスコアを抽出するためのキーワード、形式については config_template.yaml ファイルを参照
```

### 判断モデル

```yaml
judge_models:
  - model_name: string # モデルのrepo_id
    api: boolean # モデルがAPI経由で推論を行うかどうか
    provider: string # モデルのプロバイダ
```

### 評価設定

```yaml
eval_datasets_dir: string # 評価データセットを含むディレクトリ
log_dir: string # ログを保存するディレクトリ
result_dir: string # 評価結果を保存するディレクトリ
existing_eval_dir: string  # オプション、既存の結果を比較するためのディレクトリ（一部のモデルの評価を再実行しない）
inference_library: string  # 使用する推論ライブラリ、huggingfaceの場合は"hf"または"huggingface"、vllmの場合は"vllm"
```

## インストール

```bash
# 仮想環境を作成
python3 -m venv .venv
# 仮想環境を有効化
source .venv/bin/activate
# 依存関係をインストール
pip install -r requirements.txt
# バグを解決するためにfilelockをアップグレード
pip install --upgrade filelock
```

### 代替の依存関係管理

このプロジェクトは依存関係管理に`pyproject.toml`を使用しています。追加の依存関係をインストールするには：

1. `pyproject.toml`ファイルに依存関係を追加します。

```bash
poetry add <dependency>
```

2. `poetry install`を実行して環境を更新します。

## 使用方法

エンドツーエンドの評価スクリプトを実行します：

```bash
python3 scripts/evaluate_llm.py --config-name config_xxx
```

`config_xxx`を`configs`ディレクトリにある設定ファイルの名前（.yamlを除く）に置き換えてください。

出力例：
```
Start Shin Rakuda evaluation...
Processing datasets: 100%|██████████| 2/2 [00:00<00:00,  5.01it/s]
Evaluating japanese_mt_bench...
Processing models: 100%|██████████| 3/3 [00:00<00:00, 15.08it/s]
...
```

評価が完了すると、設定ファイルで指定した`result_dir`に結果と可視化が保存されます。

## 今後の課題

- [ ] Llama 3.1モデルのサポートを追加
- [ ] Huggingfaceパイプラインのサポートを改善
- [ ] VLLMを更新（適切なGPUメモリ解放をサポートする最新バージョンまで）

## 貢献方法

Shin Rakudaへの貢献を歓迎します！以下の手順で貢献できます：

1. リポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing_features`）
3. 変更をコミット（`git commit -m '素晴らしい機能を追加'`）
4. ブランチにプッシュ（`git push origin feature/amazing_features`）
5. プルリクエストを作成

テストを適切に更新し、プロジェクトのコーディング基準を遵守してください。

## トラブルシューティング

- CUDA out of memoryエラーが発生した場合は、モデル設定の`max_model_len`または`max_tokens`パラメータを減らしてみてください。
- 特定のモデルやデータセットに問題がある場合は、モデルプロバイダのドキュメントやデータセットのソースで既知の制限や要件を確認してください。
- 依存関係に問題がある場合は、正しいバージョンのPythonを使用し、必要なパッケージがすべてインストールされていることを確認してください。

さらなるヘルプが必要な場合は、GitHubリポジトリでIssueを作成してください。

## ライセンス

[MIT](https://choosealicense.com/licenses/mit/)

## 引用

研究でShin Rakudaを使用する場合は、以下のように引用してください：

```bibtex
@software{shin_rakuda,
  author = {YuzuAI},
  title = {Shin Rakuda: 柔軟なLLMベンチマーキングツール},
  year = {2024},
  url = {https://github.com/yourusername/shin-rakuda}
}
```

## 参考文献
