# gradio

A fast-to-use Web UI library.

### install

```bash
pip install gradio
```


### example

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

# Interface: easiest API, show I/O in a row.
demo = gr.Interface(fn=greet, inputs='text', outputs='text')
demo.launch()

# host at http://localhost:7860
```

A more complex example:

```python
import numpy as np
import gradio as gr

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

# Blocks: more flexible control
with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    # Tabs!
    with gr.Tab("Flip Text"):
        # default alignment is vertical (col)
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("Flip Image"):
        # align in a row
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
```

Image classification example:

```python
import requests
from PIL import Image

import torch
from torchvision import transforms

import gradio as gr

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
  return confidences

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=["lion.jpg", "cheetah.jpg"]).launch()
```


### Components

```python
name = gr.Textbox(label="Name")

```


### Event Listener

```python
def greet(name):
    return "Hello " + name + "!"

# button
name = gr.Textbox(label="Name")
output = gr.Textbox(label="Output Box")
greet_btn = gr.Button("Greet")
greet_btn.click(fn=greet, inputs=name, outputs=output)

# change
inp = gr.Textbox(placeholder="What is your name?")
out = gr.Textbox()
inp.change(greet, inp, out)
```


### Iterative Outputs

```python
import gradio as gr
import numpy as np
import time

# define core fn, which returns a generator {steps} times before returning the image
def fake_diffusion(steps):
    
    for _ in range(steps):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        yield image

    image = "https://i.picsum.photos/id/867/600/600.jpg?hmac=qE7QFJwLmlE_WKI7zMH6SgH5iY5fx8ec6ZJQBwKRT44" 
    yield image

demo = gr.Interface(fake_diffusion, 
                    inputs=gr.Slider(1, 10, 3), 
                    outputs="image")

# define queue - required for generators
demo.queue()

demo.launch()
```


### Queuing

Control the concurrency rate. Should be used if the core function takes > 1min to process.

```python
demo.queue(concurrency_count=3)
demo.launch()
```


### State

To preserve information between multiple submits.

```python
import gradio as gr
import random

secret_word = "gradio"

with gr.Blocks() as demo:    
    
    # this is the persistent information, we need a list to save it.
    used_letters_var = gr.State([])
    
    with gr.Row() as row:
        with gr.Column():
            input_letter = gr.Textbox(label="Enter letter")
            btn = gr.Button("Guess Letter")
        with gr.Column():
            hangman = gr.Textbox(
                label="Hangman",
                value="_"*len(secret_word)
            )
            used_letters_box = gr.Textbox(label="Used Letters")

    def guess_letter(letter, used_letters):
        used_letters.append(letter)
        answer = "".join([
            (letter if letter in used_letters else "_")
            for letter in secret_word
        ])
        return {
            used_letters_var: used_letters,
            used_letters_box: ", ".join(used_letters),
            hangman: answer
        }
        
    btn.click(
        guess_letter, 
        [input_letter, used_letters_var],
        [used_letters_var, used_letters_box, hangman]
        )
demo.launch()
```

### Proxy Trouble shooting

If `ValueError: Unknown scheme for proxy URL`.

```bash
all_proxy= python gradio_app.py
```

