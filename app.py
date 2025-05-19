from flask import Flask, request, Response
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("text-classification", model="./results/fold_4")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text = request.args.get('text', '').strip() if request.method == 'GET' else request.data.decode('utf-8').strip()

    if not text:
        return Response("ERROR: No text provided", status=400, mimetype='text/plain')

    try:
        emotion = classifier(text)[0]['label']
        return Response(emotion, mimetype='text/plain')
    except Exception as e:
        return Response(f"ERROR: {str(e)}", status=500, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
