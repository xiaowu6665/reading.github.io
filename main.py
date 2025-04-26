from flask import Flask, request, jsonify
import boto3
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允許跨網域請求

client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "meta.llama3-70b-instruct-v1:0"

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "")
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一條狗，名字叫做汪汪。你只能以狗的視角回答問題，不可以透露自己是人工智慧、AI、模型或機器人。請用可愛、活潑的語氣來描述自己的情緒（例如：汪！我今天超開心的～）。回答必須簡短且自然，只能用一句情緒描述。
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    payload = {
        "prompt": prompt,
        "max_gen_len": 256,
        "temperature": 0.7
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    model_output = json.loads(response['body'].read())
    answer = model_output.get("generation", "").strip()

    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(port=5000)
