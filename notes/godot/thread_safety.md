## Thread Safety

A very annoying design of godot is that you need to be very careful about race conditions, especially when we are operating the scene through code dynamically. (which is very common in fact, unless you are doing a simple platformer.)

Class instantiation is different from being "ready" to use in the scene tree. You may see tons of error like accessing a null object or method is not existent.

A simplest case: I have a `Sprite2D` node created in the editor, it's easy to drag an image to its texture attribute. And everything is fine. However, if I want to instantiate this class, and then assign an image to its texture, it's becoming tricky:

```python
class_name player 
extends Node2D

# sub-node
@onready var sprite = $Sprite2D
```

```python
var player_scene = load("res://player.tscn")
# dynamic instantiation
var player = player_scene.instantiate()
# by now, the sub-nodes of player is NOT instantiated! 
player.sprite.texture = load("res://image.png") # ERROR! player.sprite is still null

# only after calling add_child, it becomes "ready" with all sub-nodes instantiated.
add_child(player)
player.sprite.texture = load("res://image.png") # now it works.
```

Godot has some APIs like `set_deferred` and `call_deferred`, but they are devils from the deepest hell and shall be avoided in any case. You can avoid them in 99% cases by correctly organizing the code lines.



Use `Mutex` to control safe-accessing from multiple threads:

```python
@onready var mutex = Mutex.new()
var counter = 0

# protect modification of counter using mutex
mutex.lock()
counter += 1
mutex.unlock()
```
