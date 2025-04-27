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
    
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
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
