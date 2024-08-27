# Godot 4.2 Basics

### Concepts and Classes

A **Game** is a loop (`MainLoop --> SceneTree`) that iterates over time.

A `SceneTree` object is used to manage the game by loading/switching scenes.

A **Scene** is a tree of `Node` or scene instances. **Scene is not a class**! It's aimed for reuse and saved in **tscn** (text scene) files. 

* You can have multiple instances from the same class (may be extended by the same script) in a scene to use different properties.

* Another scene could also be instantiated and added to a scene. This will instantiate all the nodes inside the child scene.

`Variant` is the dynamic type to hold any kind of data.

`Object` is the base class for (almost) everything.

```bash
# Class tree of Object
Object --> RefCounted --> Resource --> Script/Texture/Shape2D...
       ⊢-> MainLoop --> SceneTree
       ⊢-> ResourceLoader
       ⊢-> Node --> CanvasItem --> Node2D --> Sprite2D/...
                |              ⊢-> Control  
                ⊢-> Node3D 
                ⊢-> Viewport --> Window 
                ⊢-> CanvasLayer
```

```python
### Object
# general method for all notifications, like _enter_tree, _exit_tree, _ready, ...
void _notification(int what) 

# call a method using its name and return its results, == method()
Variant call(StringName method, ...) # ... are args to method

# deferred call at idle time (at the end of frame)
# always return null instead of the method's results!
Variant call_deferred(StringName method, ...)

# assign value to property using func
void set(StringName property, Variant value)

# assign value to property at idle time.
# == call_deferred(set, property, value)
void set_deferred(StringName property, Variant value)
```

`RefCounted` inherits from `Object` to allow garbage-collection.

`Resource` is the base class for **Serializable** objects. (e.g., `Script`, `AudioEffect`)

`ResourceLoader` is a helper to load resource files. It's a built-in singleton.

`SceneTree` implements the game loop, and hold all scenes and nodes. It's a built-in singleton.

```python
### SceneTree
Window root # the root node! Window is the default Viewport.
Node current_scene
bool paused = false
```

`Veiwport` defines what is displayed in the screen. It's also the root node of the `SceneTree`, and all other scenes should be children of this node to be displayed.

```bash
### Scene tree
SceneTree --> root (Viewport) --> Scene --> Node
```

`CanvasLayer` is for separate rendering of objects (its children) in 2D. 

![image-20240809232549518](D:\aa\Notebooks\docs\godot\basics.assets\image-20240809232549518.png)

`Node` is the smallest building block for the game. 

It implements editable properties, callbacks for the game loop, and `add_child` to build a tree.

```python
### Node
StringName name

# constructor
void _init()
# notifications
void _enter_tree()
void _exit_tree()
void _ready()
void _process()
void _physics_process()
# input event handler
void _input(InputEvent event)
void _unhandled_input(InputEvent event)
# enable/disable processing
void set_process(bool enable)
void set_physics_process(bool enable)
void set_process_input(bool enable)
void set_process_unhandled_input(bool enable)
# check status
bool is_node_ready()
bool is_processing()
bool is_physics_processing()
bool is_in_group(StringName group)
String get_class() # return class name
# scenetree 
SceneTree get_tree() # get SceneTree
Viewport get_viewport() # get viewport, usually == get_tree().root
NodePath get_path() # get absolute path of current node
# other nodes
void add_child(Node node, ...) # add a child node
Node get_parent() # null if no parents
Node get_node(NodePath path) # get node from path. Relative path is from this node, absolute path is from SceneTree.root
bool has_node(NodePath path) # if contains a node
# actions
void queue_free() # safely delete this node at the end of frame, will delete all children too. Useful to clear a node.
```

`CanvasItem`  is the abstract base class for all 2D things, including `Node2D` and `Control`.

```python
###  CanvasItem
# visibility of 2D objects
bool visible = true
# transform
Vector2 offset = Vector2(0, 0)
float rotation = 0
Vector2 scale = Vector2(1, 1)
# drawing order
int z_index = 0 # in [-4096, 4096], higher values will be drawn forward

void hide() # visible = false
void show() # visible = true
```

`Node2D` is the base class for 2D-related objects.

`Control` is the base class for GUI objects.

`Node3D` is the base class for 3D-related objects.


### Game Logic

What happens on starting the game:

* `OS` is created to handle operating system related things.

*  `SceneTree` is created, which contains the root `Viewport`.

* Children of `Viewport` (active scenes) enter the `SceneTree`.

* Active scene load its `Node`s in the tree order (parent-to-children).
  * `Node._init()` is called to initialize the instance.
  * `Node._enter_tree()` is called before loading its children, so it's called from parent-to-child.
  * **After all Nodes are loaded**, `Node._ready()` is called. The order is therefore reversed from child to parent.
* The game loop starts, which iteratively calls:
  * `Node._process(float delta)`. This happens as fast as possible, so the actual time delta is provided. Use `Node.set_process(bool enable)` to enable/disable it. Defaults to enabled if `_process()` is implemented (overridden).
  * `Node._physics_process()` for a fixed number of times every second (e.g., 60 FPS). Use `Node.set_physics_process(bool enable)` to enable/disable it.

* The active scene may be changed to another scene:

  * this is usually done in script:

    ```python
    func _level_completed():
        # dynamic load at runtime
        get_tree().change_scene_to_file('res://level2.tscn')
    
    # or preload at compile time
    var level2 = preload('res://level2.tscn') # PackedScene
    func _level_completed():
        get_tree().change_scene_to_packed(level2)
    ```


### GDScript

`Script` is a kind of `Resource` to extend functionality of an `Object` by **attaching** to a Node.

It will **inherits (extends)** the original class as a new class to add features. Each script represents a new class definition, and is saved in **gd** file.

For each instance of a base class, we can attach different scripts (so they become different child classes) or the same scripts (the same child class).

The grammar is almost pythonic, but:

```python
### there is no need for `self`
# just call the function directly (even from parent class).
queue_free() # if inherited from Node

### literals
false, true
null

### variable declaration
var x # null by default, dynamic type
var x: int # static type (can only be int!)
var x = 1 # init to int, but still dynamic type
var x: int = 1 # static type
var x := 1 # static type (auto inferred as int)

const x = 1 # const variable
static var x # static variable (shared among instances)

# note that any static variables will cause the class to NEVER be freed! (ref: https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_basics.html#:~:text=Currently%2C%20due%20to%20a%20bug%2C%20scripts%20are%20never%20freed%2C%20even%20if%20%40static_unload%20annotation%20is%20used.)


### different strings
"xxx" # String, general use, copy-on-write
&"xxx" # StringName, immutable and unique, faster for comparison.
^"xxx" # NodePath, pre-parsed to reference a node (or its resources and properties)

### division 
5 / 2 == 2 # both operands are int, output is also int!
5 / 2.0 == 2.5 # manual float conversion

### short hand
$NodePath # == get_node("NodePath")
%UniqueNode # == get_node("%UniqueNode")

### annotations (decorators)
@export var x: int = 1 # export an integer
@export_range(1, 100, 1) # export a slider to editor
var range: int = 50
@export_file var f # string as path to file

@onready var my_label = get_node("MyLabel") # will defer initialization after ready.


### code regions (to fold code quickly, only supported in built-in code editor...)
#region Description
...
#endregion

### array
var l = [1, 2, 3]
var len = l.size()

l.append(4) # == l.push_back(4)
l.push_front(5)

l.erase(4) # remove by value
l.remove_at(0) # remove by index

var i = l.find(2) # find by value, return index
print(2 in l) # test containing by `in`
print(l.count(2)) # count by value

l.reverse()
l.shuffle()
l.clear()

### dict
var d = {"key": 0}
d.one = 1 # == d['one'] = 1
print(d["one"])
print('one' in d) # containing test by `in`

### for loop
for i in range(10): # range(s, e, i)
    print(i)
for i in [1, 2, 3]: # array
    print(i)
d = {"a": 0, "b": 1}    
for key in d: # dict
    print(d[key])
l = ["a", "b"]
for x: String in l: # typed array
    print(x)

### function (Callable)
# functions are first-class items (can be directly referenced by name, instead of by string)
# however, we need to use func.call() when func is passed as an argument:
func foo():
    print('foo')

func bar(f: Callable):
    f.call() # cannot use f(), must use f.call()
    
func _ready():
	foo() # just use the () operator is OK.
    bar()

# gdscript DO NOT support named argument!
# ref: https://github.com/godotengine/godot-proposals/issues/902
bar(f=foo) # error!

# lambda functions also need to use .call()
var foo = func(): print('foo')
foo.call() # cannot use foo()

# lambda functions will capture local env
var x = 42
var my_lambda = func(): print(x)
my_lambda.call() # Prints "42"
x = "Hello"
my_lambda.call() # Prints "42"

### named class
# by default, gdscript extends a class and will create an UNNAMED class, but we can also name it (even icon it)
@icon("res://path/to/icon.png")
class_name MyClass # after adding this line, you can found it in the editor!
extends Node # if not speficied, will default to `extends RefCounted`

# in another script, you can extend from your class too
extends MyClass
# if you don't define the class_name, you can also use path to extend
extends "res://src/MyClass.gd"

# use keyword `is` to check class
const myclass = preload("myscript.gd") # load script (class definition) only once at compile time.
if myclass is MyClass:
    pass
var myinstance = myclass.new() # instantiate a class

const myscene = preload("myscene.tscn") # PackedScene
var myscene_node = myscene.instantiate() # instantiate a scene
add_child(myscene_node) # add to current node

# use super to call parent methods
func not_overriding():
    super.not_overriding()
    
### set and get for inter-dependent properties
var milliseconds: int = 0
var seconds: int:
	get:
		return milliseconds / 1000
	set(value):
		milliseconds = value * 1000

### assert
assert(i == 0)
assert(i == 0, 'i is not 0')

### enums (a special constant)
# unnamed
enum {TILE_BRICK, TILE_FLOOR, TILE_SPIKE, TILE_TELEPORT}
# Is the same as: const TILE_BRICK = 0; const TILE_FLOOR = 1

# named
enum State {STATE_IDLE, STATE_JUMP = 5, STATE_SHOOT} # Access values with State.STATE_IDLE, etc.
# Is the same as: const State = {STATE_IDLE = 0, STATE_JUMP = 5, STATE_SHOOT = 6}

```


### User Input

User input will **propagate from-child-to-parent** through the node tree until one node consumes it. 

Each single user input (press a key, click mouse) will be propagated separately as a sub-class of `InputEvent`:

```bash
Resource --> InputEvent --> InputEventFromWindow --> InputEventWithModifiers --> InputEventKey/InputEventMouse/InputEventGesture
                                                 ⊢-> InputEventScreenDrag/InputEventScreenTouch
                        ⊢-> InputEventAction
                        ⊢-> InputEventShortcut
                        ⊢-> InputEventJoypadButton/InputEventJoypadMotion                 
```

```python
### InputEvent

# query action group press/release
bool is_action_pressed(StringName action, bool allow_echo=false, bool exact_match=false)
bool is_action_released(StringName action, bool allow_echo=false, bool exact_match=false)

# key and mouse press/release
bool is_pressed() const 
bool is_released() const

# return event name
String as_text() const

# 0-1, for joypad strength
float get_action_strength(StringName action, bool exact_match=false) const 
```

Node should override `_input()` to handle targeted input. 

`set_process_input(enable)` can be used to turn on/off input processing.

The order of input processing:

```python
# firstly, all node can catch input
Node._input()

# secondly, GUI will catch input
# GUI will not follow the child-to-parent order!
Control._gui_input() 

# then, node will catch again for InputEventKey/JoypadButton/Shortcut, no mouse input!
Node._shortcut_input()

# node will catch InputEventKey again.
Node._unhandled_key_input()

# finally...
Node._unhandled_input()
```

#### Event and Poll

**Event**: input that only respond once, e.g., click mouse, press enter.

We should use the `_input()` system and `InputEvent` class to handle events:

```python
func _input(event):
    if event.is_action_pressed("ui_left"):
        # turn left
```

**Poll**: input that happens as long as it's pressed. e.g., press left key to move until released.

We need to constantly check the status in `_process()`, which can be done by a built-in Singleton `Input`.

```python
func _physics_process(delta):
    if Input.is_action_pressed("ui_left"):
        # move left
        position.x += speed * delta
```

#### Mouse

`InputEventWithModifiers`: modifier means keys like shift or alt, it can distinguish `Shift+Click` from `Click`.

`InputEventMouse` add `position` and `global_position` of mouse. Further it's inherited by:

`InputEventMouseButton` for button click / wheel scroll.

```python
MouseButton button_index
bool double_click
bool pressed
```

```python
func _input(event):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			print("Left button was clicked at ", event.position)
		if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			print("Wheel up")
```

`InputEventMouseMotion` for drag.

```python
float pressure
Vector2 relative
Vector2 velocity
```

```python
var dragging = false
var click_radius = 32 # Size of the sprite.

func _input(event):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
        # note that even if the mouse is not on your node, it will still get propagated! 
        # we must detect if mouse is on our sprite.
		if (event.position - $Sprite2D.position).length() < click_radius:
			# Start dragging if the click is on the sprite.
			if not dragging and event.pressed:
				dragging = true
		# Stop dragging if the button is released.
		if dragging and not event.pressed:
			dragging = false

	if event is InputEventMouseMotion and dragging:
		# While dragging, move the sprite with the mouse.
		$Sprite2D.position = event.position
```

#### Touch

`InputEventScreenTouch` is similar to `InputEventMouseButton`.

`InputEventScreenDrag` is similar to `InputEventMouseMotion`.

To test touch input using PC, enable `Project settings > Input Devices/Pointing > Emulate Touch From Mouse`

#### Actions (Keyboard & Controller)

For keyboard/Joypad input, we majorly use `InputEventAction` through `InputMap`, which is more flexible:

* same code for different devices: left key & joypad push to left.
* runtime reconfiguration.
* runtime programmatic triggering.

To find all predefined user input (ui) names: `Project --> Project Settings --> Input Map (check Show built-in actions)`

Some commonly used input map names:

* `ui_accept`: Enter, Space
* `ui_cancel`: Escape
* `ui_left/right/up/down`: Left/Right/Up/Down keys.

Although `InputEventAction` is recommended, sometime you may want to use the `InputEventKey` specially for keyboard:

```python
func _input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_T:
			if event.shift_pressed:
				print("Shift+T was pressed")
			else:
				print("T was pressed")
```

#### Quit request

By default, when the window is force closed, godot will send `NOTIFICATION_WM_CLOSE_REQUEST` and auto quit safely.

To send your own quit notification (e.g., in-game button to quit to desktop), use

```python
# notify and allow actions to finish (e.g., saving)
get_tree().root.propagate_notification(NOTIFICATION_WM_CLOSE_REQUEST)

# quit safely
get_tree().quit()
```


### Signals

Signals are used to communicate between Nodes.

`Signal` is a variant class since Godot 4.0.

We can use the editor GUI to connect signals: `Node --> Signals` panel to see all the signals that a Node can send.

Double-click a signal to **connect** it to another Node.

Then, the another Node needs to implement a receiver function to catch the signal, which will be automatically created on connection, called `_on_<node name>_<signal name>()`.

e.g., toggle processing when press a button:

```python
func _on_button_pressed():
    set_process(not is_processing())
```

We can also use script to connect signals.

```python
### script of the node that receives signal

# do connection after all nodes are ready.
func _ready():
    # get reference to the node that sends signal
    var timer = get_node("Timer") 
    # timeout is the signal name
    timer.timeout.connect(_on_timer_timeout)

# the receiver function
func _on_timer_timeout():
    visible = not visible
```

We can create **custom signals** using script.

```python
### script of the node that sends signal
extends Node2D

signal health_depleted # use signal keyword
signal health_changed(old_value, new_value) # signal can also have arguments!
var health = 10

func take_damage(amount):
    var old_health = health
    health -= amount
    if health <= 0:
        health_depleted.emit() # use emit() method to send 
    health_changed.emit(old_health, health) # emit with parameters
```

use coroutine to wait for signals:

```python
func wait_confirmation(): # becomes a coroutine
    await $Button.button_up # will pause until receives the signal
    return true

func _ready():
    var confirmed = await wait_confirmation() # will pause
    wait_confirmation() # won't pause!
```


### Singletons (Autoload)

Singletons is a special Node that is always loaded, and can be referenced by all other Nodes.

It is usually used to store **global** information used by multiple scenes.

`SceneTree` can also be viewed as a singleton.

Use the editor `Project > Project Settings > Autoload` to declare a GDScript file (enter path or browse folder) as a singleton, and also **assign a name**.

Then this script's variables can be referenced using its name in all other scripts!

For example, we add a singleton named  `Global`:

```python
### global.gd
extends Node

var current_scene = null;

# we still need to fetch other things in ready
func _ready():
    var root = get_tree().root
    current_scene = root.get_child(root.get_child_count() - 1)

# use singleton to manage scene changing
func goto_scene(path):
    # deferred call to avoid racing
    call_deferred("_deferred_goto_scene", path)

func _deferred_goto_scene(path):
    # deferred call is safe to free
    current_scene.free()
    # load a scene
    var s = ResourceLoader.load(path) # PackedScene
    #var s = load(path) # same, load is an alias of ResourceLoader.load
    current_scene = s.instantiate() # Node
    get_tree().root.add_child(current_scene) # activate it
    get_tree().current_scene = current_scene
```

Then we can use `Global` singleton in other scripts:

```python
### level1.gd

func _level_completed():
    # use Global to change scene
    Global.goto_scene("res://level2.gd")
```


### Physics

2D physics is basically about collision detection & response.

Four types of collision objects,  with inheritance tree as:

```bash
# node
Node2D --> CollisionObject2D --> Area2D
       |                     ⊢-> PhysicsBody2D --> StaticBody2D
       |                                       ⊢-> RigidBody2D
       |                                       ⊢-> CharacterBody2D
       ⊢-> CollisionShape2D
       ⊢-> CollisionPolygon2D

# resource
Resource --> PhysicsMaterial
         ⊢-> Shape2D
```

`CollisionObject2D` is an abstract class for 2D physics, which implements:

```python
int collision_layer = 1 # appear in what layers
int collision_mask = 1 # collide with what layers

signal mouse_entered
signal mouse_exited

void _mouse_enter() virtual
void _mouse_exit() virtual
```

Collision layer and mask can be used to setup which objects should collide with which objects. 

For example, we want mobs to collide with player (both check layer 1), but mobs should not collide with other mobs (uncheck mask 1 for mobs).

We can assign names to layers in `Project Settings --> Layer Names` for easy memorization.


`Area2D` ("area") implements **detection and influence** from/to other `CollisionObject2D`. However, **area is not intended for physics simulation**.

For example: 

* projectile that deals damage (can be detected), but triggers no physics (bouncing)
* "detect" area, so enemy can see everything enters the area.
* change of physics laws in this area (e.g., gravity).

```python
Area2D[] get_overlapping_areas ( ) const
Node2D[] get_overlapping_bodies ( ) const
bool has_overlapping_areas ( ) const
bool has_overlapping_bodies ( ) const
bool overlaps_area ( Node area ) const
bool overlaps_body ( Node body ) const
```

`PhysicsBody2D` ("body") **implements physics simulation**. **Body is usually not intended for detection.**

For example, `StaticBody2D` implements static objects (environment, like a wall, a platform).  `RigidBody2D` implements rigid body with a fixed shape (regular enemy, the sprite may animate, but collision box is rigid). `CharacterBody2D` implements detection, and is used for controllable characters.


To correctly set up physics, we need to build node trees and connect signals correctly:

`CollisionObject2D` should be **be parent of `CollisionShape2D` /`CollisionPolygon2D`  to define collision area**:

```python
# player scene, can detect and influence
Player (extends Area2D)
⊢-> Sprite2D 
⊢-> CollisionShape2D 
    .shape = Shape2D 

# mob scene, can only influence
Mob (extends RigidBody2D)
⊢-> Sprite2D 
⊢-> CollisionShape2D 
    .shape = Shape2D 
    
# Player should detect any collision from other body, so we connect the body_entered signal from Area2D and impl:
signal hit # further send out a signal
func _on_body_entered(body):
	hide() # Player disappears after being hit.
	hit.emit()
	# Must be deferred as we can't change physics properties on a physics callback.
	$CollisionShape2D.set_deferred("disabled", true)

```

`CollisionShape2D` /`CollisionPolygon2D` should contain a `Shape2D` / polygon (`PackedVector2Array`) that defines the 2D shape:

```python
# CollisionShape2D
Shape2D shape
bool disabled = false
bool one_way_collision = false
# CollisionPolygon2D
PackedVectro2Array polygon
bool disabled = false
bool one_way_collision = false
```

`Shape2D` is an abstract class for 2D shape `Resource` (not `Node`! it won't show in the node editor, but show as a property of `CollisionShape2D` /`CollisionPolygon2D`  in the inspector). 


### Math

2D Coordinate system:

```bash
(0, 0) ------- +x
|
|
|
+y
```

```python
# vector 2
var v = Vector2(2, 4)
print(v.x, v.y)

v = v.normalized()

# matrix 2x2
var t = Transform2D()
print(v.x.x, v.y.x,
      v.x.y, v.y.y) # colomn-major!
```

3D Coordinate system: OpenGL convention.


### Importing Assets

Images will be imported as `Texture` by default.

To ensure high-quality image, we have two approaches:

* Use a high base resolution (like 4K = 3840 x 2160) and high resolution texture images, then we can always mipmap to lower resolution if needed.
* Use a normal base resolution (1920 x 1080) and high resolution texture images, and adjust the `scale` to down-scale it.

Usually we prefer the first approach, as it doesn't need to calculate scales.

### Groups

Groups are a convenient container for nodes, which allows to:

- Get a list of nodes in a group.
- Call a method on all nodes in a group.
- Send a notification to all nodes in a group.

In the editor, you can manage groups manually in `right panel > nodes > groups`.

Or use script to manage groups:

```python
# add current node to group
add_to_group("guards") # will create the group automatically

# call method
get_tree().call_group("guards", "enter_alert_mode")

# get all nodes in a group
var guards = get_tree().get_nodes_in_group("guards")
```

### Tween Animation

`Tween` is inherited from `RefCounted`, and is used for interpolation-based animation (in-betweening), e.g., moving the position smoothly.

Tween animations are played after `_process()`.

Tweens are not designed to be reused, always create one when needed:

```python
var tween;

# must use this way to create!
tween = create_tween() 

# zoom up by 2 in 1 second (object, property, final_val, duration)
tween.tween_property($Sprite, "scale", Vector2(2, 2), 1) # this is called a tweener

# idle for 1 second interval
tween.tween_interval(1)

# call a method after 1 second
tween.tween_callback(myfunc).set_delay(1)

# call a method in the way of property
func myfunc(x, y): ...
tween.tween_method(myfunc.bind(Y), X0, X1, duration) # use bind(Y) to set a default y
```

Tweeners will be executed one-after-one by default. To parallel-tween multiple objects or multiple properties, use:

```python
var tween = create_tween()
tween.tween_property(...)
tween.parallel().tween_property(...) # parallel with the first one
tween.parallel().tween_property(...) # parallel with the second one (and the first one)

### or set_parallel by default
var tween = create_tween()
tween.set_parallel()
# all later tweeners  will be parallel 
```

However, you should **not allow multiple tweeners on the same property** at the same time, this will be undefined behavior (race condition).


### Thread Safety

Interacting with the active scene is not thread-safe, always use deferred call:

```python
# Unsafe:
node.add_child(child_node)
# Safe:
node.call_deferred("add_child", child_node)
```

Use `Mutex` to control safe-accessing from multiple threads:

```python
@onready var mutex = Mutex.new()
var counter = 0

# protect modification of counter using mutex
mutex.lock()
counter += 1
mutex.unlock()
```


Signals?


### Editor tips

* When resize something in the scene viewer:
  * `Shift` to keep aspect ratio.
  * `Control` to snap to grid
  * `Alt` and scale the 2D shapes so the anchor is centered.
