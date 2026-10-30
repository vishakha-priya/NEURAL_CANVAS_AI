import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
Base_Dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__,
            template_folder=os.path.join(Base_Dir,"templates"),
            static_folder=os.path.join(Base_Dir,"static"))
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(
    Base_Dir, 'static', 'uploads'
)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VGG_PATH = os.path.join(BASE_DIR, "vgg_normalised.pth")
DECODER_PATH = os.path.join(
    BASE_DIR, "experiment", "final_exp", "decoder_final.pth"
)

encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(DECODER_PATH, map_location=device)
)

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)




@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    # Restore previously uploaded filenames from hidden fields.
    content_filename = form.content_path.data or None
    style_filename = form.style_path.data or None
    result_image = None
    error = None

    if form.validate_on_submit():
        try:
            # Save a newly uploaded content image, if provided.
            if form.content.data and form.content.data.filename:
                if not allowed_file(form.content.data.filename):
                    raise ValueError(
                        "Invalid content image format. Use PNG, JPG, or JPEG."
                    )

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], content_filename
                )
                form.content.data.save(content_path)
                form.content_path.data = content_filename

            # Save a newly uploaded style image, if provided.
            if form.style.data and form.style.data.filename:
                if not allowed_file(form.style.data.filename):
                    raise ValueError(
                        "Invalid style image format. Use PNG, JPG, or JPEG."
                    )

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], style_filename
                )
                form.style.data.save(style_path)
                form.style_path.data = style_filename

            # Check that both images exist.
            if not content_filename or not style_filename:
                raise ValueError(
                    "Please upload both content and style images."
                )

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'], content_filename
            )
            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'], style_filename
            )

            if not os.path.isfile(content_path):
                raise FileNotFoundError(
                    f"Content image not found: {content_filename}"
                )

            if not os.path.isfile(style_path):
                raise FileNotFoundError(
                    f"Style image not found: {style_filename}"
                )

            # Load images.
            with Image.open(content_path) as img:
                content_image = img.convert('RGB')

            with Image.open(style_path) as img:
                style_image = img.convert('RGB')

            # Run AdaIN style transfer.
            alpha = float(form.alpha.data)

            stylized_image = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                alpha,
                device
            )

            # Save the generated image.
            result_filename = 'stylized_' + content_filename
            result_path = os.path.join(
                app.config['UPLOAD_FOLDER'], result_filename
            )

            save_image(stylized_image, result_path)

            if not os.path.isfile(result_path):
                raise FileNotFoundError(
                    "The stylized output image was not saved."
                )

            result_image = result_filename

        except Exception as e:
            app.logger.exception("Style transfer failed")
            error = str(e)

    elif request.method == 'POST':
        # Preserve the filenames even if form validation fails.
        error = "Form validation failed. Please check the uploaded files and try again."

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    example_dir=os.path.abspath(os.path.join(app.root_path,'..','examples'))
    return send_from_directory(example_dir, filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
