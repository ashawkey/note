# DearPyGUI

### Install

```bash
pip install dearpygui
```



### Pipeline

Context --> Viewport --> DearPyGUI

```python
import dearpygui.dearpygui as dpg

# Create context
dpg.create_context()

# Define the primary window
with dpg.window(label="Primary Window"):
    # Add items
    dpg.add_text("Hello, world")
    dpg.add_button(label="Save")
    dpg.add_input_text(label="string", default_value="Quick brown fox")
    dpg.add_slider_float(label="float", default_value=0.273, max_value=1)

# Create Viewport
dpg.create_viewport(title='Custom Title', width=600, height=200)

# Start DearPyGUI
dpg.setup_dearpygui()
dpg.show_viewport()

# Render Loop
while dpg.is_dearpygui_running():
    print("this will run every frame")
    dpg.render_dearpygui_frame()

# Destroy Context
dpg.destroy_context()
```

