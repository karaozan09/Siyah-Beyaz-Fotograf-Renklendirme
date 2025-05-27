from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from colorizers import *

# Dosya yükleme dizinini ayarla
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Yüklenebilir dosyalar için kontrol
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Dosya kontrolü
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('process_image', filename=filename))
    return render_template('upload.html')

@app.route('/process/<filename>')
def process_image(filename):
    # Burada renk dönüştürme işlemlerini yap
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    colorized_images = process_and_colorize_image(img_path)

    return render_template('display.html', 
                         original_image=filename,
                         colorized_images=[os.path.basename(path) for path in colorized_images])

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_and_colorize_image(img_path):
    # Yüklenen resmi renkli hale getirme işlemleri
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()

    img = load_img(img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Sonuçları kaydet
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    eccv16_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_eccv16.png")
    siggraph17_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_siggraph17.png")

    plt.imsave(eccv16_path, out_img_eccv16)
    plt.imsave(siggraph17_path, out_img_siggraph17)

    return [eccv16_path, siggraph17_path]

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)