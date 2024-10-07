import logging
import subprocess
from typing import List, Self

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig, AutoTokenizer, AutoModel

# 1. BigBirdのトークナイザーとモデルをロード
tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/bigbird-base-japanese")
config = AutoConfig.from_pretrained(
    "nlp-waseda/bigbird-base-japanese", attention_type="original_full"
)
model = AutoModel.from_pretrained(
    "nlp-waseda/bigbird-base-japanese", config=config
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


class Body(BaseModel):
    inputText: str = Field(examples=["春は曙、夏は夜、秋は夕暮れ、冬は夜更け"])


class UserRequestIn(BaseModel):
    body: Body
    modelId: str = Field(examples=["amazon.titan-embed-text-v1"])
    accept: str = Field(examples=["application/json"])
    contentType: str = Field(examples=["application/json"])

    @model_validator(mode="after")
    def validate_root(self) -> Self:
        if (
            self.modelId != "amazon.titan-embed-text-v1"
            and self.modelId != "amazon.titan-embed-text-v2"
        ):
            raise ValueError(
                "modelId must be 'amazon.titan-embed-text-v1 or be amazon.titan-embed-text-v2'"
            )
        if self.accept != "application/json":
            raise ValueError("accept must be 'application/json'")
        if self.contentType != "application/json":
            raise ValueError("contentType must be 'application/json'")
        return self


class VectorizedText(BaseModel):
    embedding: List[float]


def _tokenize(text: str) -> str:
    return subprocess.run(
        ["jumanpp", "--segment"], input=text, encoding="utf-8", stdout=subprocess.PIPE
    ).stdout


@app.post("/vectorize")
def vectorize(user_request_in: UserRequestIn):
    sentence = _tokenize(user_request_in.body.inputText)
    logger.debug(sentence)
    # BigBird用のトークナイズ
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():  # 勾配計算をオフにする
        outputs = model(**inputs)

    # 4. 出力結果
    # outputs.last_hidden_state: トークンごとのベクトル表現
    # outputs.pooler_output: 文全体のベクトル表現
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    # トークンごとのベクトル表現の形状を確認（例: 1つの文、10のトークン、1,024次元の埋め込みベクトル）
    logger.info("Last Hidden State Shape: %s", str(last_hidden_state.shape))

    # 文全体のベクトル表現の形状を確認（例: 1つの文、1,024次元のベクトル）
    logger.info("Pooler Output Shape: %s", str(pooler_output.shape))

    return VectorizedText(embedding=pooler_output.tolist()[0])
