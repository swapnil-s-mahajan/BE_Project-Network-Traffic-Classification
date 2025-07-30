from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['datafile']
        # Handle file processing here
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Handle model training here
        return render_template('train.html', progress=80)  # Just an example, set actual progress here
    return render_template('train.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
