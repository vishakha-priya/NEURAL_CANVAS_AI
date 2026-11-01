##Live WEBSITE LINK
[[https://neural-canvas-ai-2.onrender.com](https://neural-canvas-ai-2.onrender.com)]

## NEURAL_STYLE_TRANSFER
I developed an AI-powered Neural Style Transfer web application using Adaptive Instance Normalization, PyTorch, and Flask.

The application takes two inputs: a content image and a style image. I used a pretrained VGG encoder to extract feature representations from both images. Then, I applied Adaptive Instance Normalization to align the statistical properties of the content features with those of the style features.

The transformed features are passed through a trained decoder to generate the final stylized image. I also implemented an alpha parameter that allows users to control the intensity of style transfer.

For the web application, I used Flask to handle image uploads, model inference, and output generation, while HTML and CSS provide the frontend interface.

The main advantage of my project is that it supports arbitrary style transfer, allowing users to apply different artistic styles without training a separate model for each style.
