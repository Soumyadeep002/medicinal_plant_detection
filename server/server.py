from flask import Flask, request, jsonify, render_template
from werkzeug import security
import util
import os

UPLOAD_FOLDER = 'E:\\My Projects\\Medicinal_plant_detection\\server\\static'

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# @app.route("/myapi", methods = ['GET'])
# def hello():
#     d = {}
#     path = str(request.args['image'])
#     output = util.classify(path)
#     d["class"] = output
#     return d

@app.route("/", methods = ['GET', 'POST'])
def upload_media():
    return render_template('index.html')

@app.route("/classify", methods = ['GET', 'POST'])
def classify():
    if (request.method == 'POST'):
        f = request.files['upload']
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
        path = str(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
        output = util.classify(path)
        return render_template('output.html', prediction = output)

if __name__ == "__main__":
    app.run(port=5500, debug = True)