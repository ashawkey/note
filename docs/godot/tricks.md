# Pitfalls & Tricks

### Panel

`Panel` is used to represent a box-like UI, e.g.

* background panel for other UI elements.
* a round-corner boundary, to show "highlight" effect.

To change the look, we need to create a new `StyleBox` inside `Inspector > Theme Overrides > Styles > Panel`

![image-20240813212521766](tricks.assets\image-20240813212521766.png)


### Mouse_entered/exited

Note that `mouse_exited` will be triggered if the mouse enter a child `Control/CollisionObject2D` element!

To avoid this, we need to set `mouse filter` in Inspector:

![image-20240813212853329](tricks.assets/image-20240813212853329.png)

* Stop (default): not applying filter, i.e., it will trigger mouse signals normally.
* Ignore: ignore all mouse signals.