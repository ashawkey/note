## Godot 4.2

### Concepts

A **Game** is a loop (`MainLoop --> SceneTree`) that iterates over time.

A `SceneTree` object is used to manage the game by loading/switching scenes.

A **Scene** is a tree of `Node`s. **Scene is not a class**! It's aimed for reuse and saved in **tscn** (text scene) files. 

* The Nodes in a scene are the **class** themselves, not instances.

* However, **An instance of a scene** could also be added to another scene. This will instantiate all the nodes inside the child scene, to allow for different properties.


`Variant` is the dynamic type to hold any kind of data.

`Object` is the base class for (almost) everything.

```bash
# Class tree of Object
Object --> RefCounted --> Resource --> Script/PackedScene/...
       ⊢-> MainLoop --> SceneTree
       ⊢-> ResourceLoader
       ⊢-> Node --> CanvasItem --> Node2D/Control/...
                ⊢-> Node3D 
                ⊢-> Viewport --> Window 
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
```

`RefCounted` inherits from `Object` to allow garbage-collection.

`Resource` is the base class for **Serializable** objects. (e.g., `Script`, `AudioEffect`)

`ResourceLoader` is a helper to load resource files. It's a built-in singleton.


`SceneTree` implements the game loop, and hold all scenes and nodes. It's a built-in singleton.

```python
### SceneTree
Window root # the root node!
Node current_scene
bool paused = false
```

`Veiwport` defines what is displayed in the screen. It's also the root node of the `SceneTree`, and all other scenes should be children of this node to be displayed.

```bash
### Scene tree
SceneTree --> root (Viewport) --> Scene --> Node
```


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
# enable/disable processing
void set_process(bool enable)
void set_physics_process(bool enable)
# check status
bool is_node_ready()
bool is_processing()
bool is_physics_processing()
bool is_in_group(StringName group)
# child/group/scenetree
void add_child(Node node, ...)
void add_to_group(StringName group, bool persistent=false)
SceneTree get_tree() # get SceneTree
NodePath get_path() # absolute path 
Node get_node(NodePath path) # can also use relative path
Node get_parent() # null if no parents
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

It will **inherits (extends)** the original class as a new class to add features.

GDscript are saved in **gd** files.

The grammar is almost pythonic, but:

```python
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
l.append(4)

### dict
var d = {"key": 0}
d.one = 1 # == d['one'] = 1
print(d["one"])

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
class_name MyClass
extends Node # if not speficied, will default to `extends RefCounted`

# use keyword `is` to check class
const myclass = preload("myscript.gd") # load script (class definition) only once at compile time.
if myclass is MyClass:
    pass
var myinstance = myclass.new() # instantiate

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
```


### User Input

Project --> Project Settings --> Input Map (check Show built-in actions)

Some commonly used input names:

* `ui_accept`: Enter, Space
* `ui_cancel`: Escape
* `ui_left/right/up/down`: Left/Right/Up/Down keys.


To handle an input, we need to catch them in `_process()`:

```python
if Input.is_action_pressed('ui_left'):
    direction = 'left'
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


### Best practices

##### Node tree structure

```bash
Node Main (main.gd)
* Node2D/Node3D World (world.gd)
* Control GUI (gui.gd)
```

