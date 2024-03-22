from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import subprocess
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_main_script(video_path):
    virtual_env_activate_cmd = r"sai/Scripts/activate"
    script_path = r"allmodules.py"
    
    # Activate virtual environment
    subprocess.run([virtual_env_activate_cmd], shell=True)

    # Run the script
    subprocess.run([sys.executable, script_path, video_path], shell=True)

def read_result_file():
    #result_file_path = 'gameplay_sentences.txt'
    result_file_path = 'summary24.txt'
    try:
        with open(result_file_path, 'r') as file:
            content = file.read().replace('\n', ' ')
        return content
    except FileNotFoundError:
        return 'Result file not found.'




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'cricket_video.mp4')
        file.save(filename)
        run_main_script(filename)
        return redirect(url_for('index'))  # Redirect to the 'index' route
    else:
        return 'Invalid file format! Please upload an MP4 file.'

@app.route('/get_summary')
def get_summary():
    result_content = read_result_file()
    return jsonify(result_content)

if __name__ == '__main__':
    app.run(debug=True)
