# blender tutorial

### How to rotate camera/light by parenting

* Add (`Shift + A`) a new empty object (plain axis), set its location to (0, 0, 0).
* In 3D viewport, first select camera/light (child), then select the empty object (parent).
* `Ctrl + P`, set parent to object.
* You could see in Outlier, the camera/light now belongs to the empty object.
* Rotate the empty object, and the child will rotate too!

### Render

`0` to toggle camera view. 

`F12` to render current image.

`Ctrl + F12` to render animation.

Set output path:

![image-20230305191231538](blender_tutorial.assets/image-20230305191231538.png)



### Animation

In animation tab, set keyframe by clicking:

![image-20230305191530471](blender_tutorial.assets/image-20230305191530471.png)

### Ambient lighting

In the world properties:

![image-20230305191716034](blender_tutorial.assets/image-20230305191716034.png)

### UI and shortcut key

* Middle Button (view control)
  * `Middle Button + Drag`: rotate
  * `Alt + Middle Button + Drag`: rotate to nearest axis
  * `Ctrl + Middle Button + Drag (Vertical)` == `Middle Button Scroll`: zoom
  * `Shift + Middle Button + Drag`: pan

* Numpad (view control)
  
  * `2 / 4 / 6 / 8`: rotate 
  * `1 / 3 / 7 / 9`: rotate to axis
  * `5`: toggle perspective / orthogonal
  * `0`: toggle camera view
  
  
  
* Shift + A: add new objects
  
* Shift + Z: toggle wireframe
  
* Ctrl + Space: toggle maximize area with the current application window.
  
   
  
* Ctrl + Dragging: discrete steps

* Shift + Dragging: precise steps
  
  
  
* Ctrl + Z: undo

* Ctrl + Shift + Z: redo