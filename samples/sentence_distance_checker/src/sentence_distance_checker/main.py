import json
from typing import List

import boto3
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity


class ReadableString(str):
    def read(self):
        return self


# テキストリストを入力としてAmazon Titanを用いたVectorize関数
def vectorize_document(document: str) -> List[float]:
    # # Bedrockサービスにアクセスするためのクライアントを初期化
    # client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    # # 文書をモデルに送信し、埋め込みベクトルを取得
    # response = client.invoke_model(
    #     body=json.dumps({'inputText': document}),
    #     modelId='amazon.titan-embed-text-v1',
    #     accept='application/json',
    #     contentType='application/json',
    # )

    http_response = requests.post(
        "http://localhost:9000/vectorize",
        json={
            "body": {"inputText": document},
            "modelId": "amazon.titan-embed-text-v1",
            "accept": "application/json",
            "contentType": "application/json",
        },
        timeout=30,
    )

    response = {"body": ReadableString(http_response.text)}
    print(http_response.status_code)
    if http_response.status_code != 200:
        raise Exception(
            f"Failed to get vector from Titan. Response: {http_response.text}"
        )
    # 応答から埋め込みベクトルを抽出
    response_body = json.loads(response["body"].read())
    vector = response_body.get("embedding")

    return vector


text1 = "昔々、あるところに、おじいさんとおばあさんが住んでいました。おじいさんは山へ柴刈りに、おばあさんは川へ洗濯に行きました。おばあさんが川で洗濯をしていると、大きな桃がどんぶらこ、どんぶらこと流れてきました。おばあさんはその桃を家に持ち帰り、おじいさんと一緒に食べようとしましたが、桃を割ると中から元気な男の子が出てきました。その子は桃から生まれたので、「桃太郎」と名付けられました。"
text2 = "昔々、浦島太郎という心の優しい若者がいました。ある日、浦島太郎が浜辺を歩いていると、子どもたちが亀をいじめているのを見つけました。浦島太郎は亀を助け、その亀を海に返してあげました。数日後、浦島太郎が漁に出ると、助けた亀が現れ、「竜宮城へ案内します」と言いました。浦島太郎は亀の背中に乗り、海の底にある竜宮城へと向かいました。"
text3 = "昔々、あるところに貧しいけれども心優しいおじいさんが住んでいました。ある冬の日、おじいさんは雪の中で一羽の鶴が罠にかかっているのを見つけました。おじいさんはかわいそうに思い、その鶴を助けてやりました。その晩、美しい若い娘が、おじいさんの家を訪ねてきて、「どうかしばらく泊めてください」と頼みました。おじいさんは快く娘を家に迎え入れました。"
text4 = "昔々、あるところにおじいさんとおばあさんが住んでいました。ある日、おじいさんが畑で働いていると、いたずら好きの狸が作物を荒らし始めました。おじいさんは怒って狸を捕まえましたが、狸はおばあさんをだまして逃げてしまいました。その後、狸はおばあさんにひどいことをしてしまい、おじいさんは大変悲しみました。そこへ、うさぎが現れて、おじいさんのために狸に復讐することを誓いました。"
#text5 = "昔々、足柄山に金太郎という力持ちの男の子が住んでいました。金太郎は大きな斧を持ち、山で熊や鹿などの動物たちと友達になり、いつも一緒に遊んでいました。ある日、金太郎は山奥で大きな岩を軽々と持ち上げ、その力を見せつけました。金太郎はその強さで、山の動物たちのリーダーとして尊敬されるようになりました。"
text5 = "ルパン三世は、先祖であるアルセーヌ・ルパンから引き継いだ怪盗の才能を持つ天才的な泥棒です。彼は世界各地でさまざまな宝物を盗む一方で、その人間味あふれる性格とユーモア、時には正義感から、観客に愛されるキャラクターです。ルパンは、変装や巧妙なトリックを駆使して狙った獲物を手に入れますが、その過程でしばしば予期しないトラブルに巻き込まれます。"

vector1 = vectorize_document(text1)
vector2 = vectorize_document(text2)
vector3 = vectorize_document(text3)
vector4 = vectorize_document(text4)
vector5 = vectorize_document(text5)
#vector6 = vectorize_document(text6)

vectors = np.array([vector1, vector2, vector3, vector4, vector5])
cosine_similarities = cosine_similarity(vectors)
print(cosine_similarities)
