from flask import Flask, request, send_file
import os
from pipe import runPre

app = Flask(__name__)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    front = request.files['front']
    back = request.files['back']

    front_path = os.path.join(UPLOAD_DIR, front.filename)
    back_path = os.path.join(UPLOAD_DIR, back.filename)

    front.save(front_path)
    back.save(back_path)

    # Run your grader
    runPre(front_path, back_path)

    # Send PDF back
    return send_file(r"F:\STORAGE\School\Coin_Project\gitRepo\MorganSilverDollar\Morgan_Dollar_main\test.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)