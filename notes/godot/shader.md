# Shader

There are different shader types:

* 2D: `canvas_item` 
* 3D: `spatial`
* Particle: `particles`



## 2D Shader (CanvasItem Shader)

Custom Shader scripts are stored in `*.gdshader` files, using `GDShaere` DSL.

To use custom shader, create it from inspector:

`CanvasItem > Material > New ShaderMaterial > New Shader`

This will save a `gdshader` file, so we can edit it for custom shader effects.

A CanvasItem Shader looks like:

```glsl
shader_type canvas_item;

void vertex() {
	// Called for every vertex the material is visible on.
}

void fragment() {
	// Called for every pixel the material is visible on.
}

//void light() {
	// Called for every pixel for every light affecting the CanvasItem.
	// Uncomment to replace the default light processing function with this one.
//}

```

Language basics:

```glsl
// types
void
bool bvec2 bvec3 bvec4
int ivec2 ivec3 ivec4    
float vec2 vec3 vec4 // access by .xyzw or .rgba or .stpq
uint uvec2 uvec3 uvec4    
mat2 mat3 mat4 // col-major! m[col][row]
    
// built-ins
COLOR = vec4(1,1,1,1); // the output should be written to COLOR
COLOR = vec4(UV, 0.5, 1); // UV is vec2, varies from (0, 0) to (1, 1) from left-top to right-bottom.
COLOR = texture(TEXTURE, UV); // read from default texture (e.g., sprite2d)
VERTEX += vec2(cos(TIME)*100.0, sin(TIME)*100.0); // VERTEX is the position of the draw center. TIME is just time.

// uniforms
uniform float blue = 1.0; // constant variables (uniforms)
maerial.set_shader_parameter("blue", 2.0) // set uniforms from gdscript   
    
// consts (slightly faster than uniforms, but cannot be configured)    
const float PI = 3.14159265358979323846;
    
//// Implicit type casting is not allowed!
float a = 2; // invalid
float a = 2.0; // valid
float a = float(2); // valid

uint a = 2; // invalid, 2 is default to signed int.
uint a = 2u; // valid
uint a = uint(2); // valid

//// construct vectors
vec4 a = vec4(0.0, 1.0, 2.0, 3.0);
vec4 a = vec4(vec2(0.0, 1.0), vec2(2.0, 3.0));
vec4 a = vec4(vec3(0.0, 1.0, 2.0), 3.0);
vec4 a = vec4(0.0); // (0.0, 0.0, 0.0, 0.0)

mat2 m2 = mat2(vec2(1.0, 0.0), vec2(0.0, 1.0));
mat3 m3 = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
mat4 identity = mat4(1.0);

mat3 basis = mat3(MODEL_MATRIX);
mat4 m4 = mat4(basis); // small-to-big: others set as an Identity matrix!
mat2 m2 = mat2(m4); // big-to-small: truncate left-top submatrix

// swizzling
vec4 a = vec4(0.0, 1.0, 2.0, 3.0);
vec3 b = a.rgb; // Creates a vec3 with vec4 components.
vec3 b = a.ggg; // Also valid; creates a vec3 and fills it with a single vec4 component.
vec3 b = a.bgr; // "b" will be vec3(2.0, 1.0, 0.0).
vec3 b = a.xyz; // Also rgba, xyzw are equivalent.
vec3 b = a.stp; // And stpq (for texture coordinates).
float c = b.w; // Invalid, because "w" is not present in vec3 b.
vec3 c = b.xrt; // Invalid, mixing different styles is forbidden.
b.rrr = a.rgb; // Invalid, assignment with duplication.
b.bgr = a.rgb; // Valid assignment. "b"'s "blue" component will be "a"'s "red" and vice versa.

// precision
lowp vec4 a = vec4(0.0, 1.0, 2.0, 3.0); // low precision, usually 8 bits per component mapped to 0-1
mediump vec4 a = vec4(0.0, 1.0, 2.0, 3.0); // medium precision, usually 16 bits or half float
highp vec4 a = vec4(0.0, 1.0, 2.0, 3.0); // high precision, uses full float or integer range (32 bit default)

// array (c-like)
float arr[3];
float float_arr[3] = float[3] (1.0, 0.5, 0.0); // first constructor
int int_arr[3] = int[] (2, 1, 0); // second constructor
vec2 vec2_arr[3] = { vec2(1.0, 1.0), vec2(0.5, 0.5), vec2(0.0, 0.0) }; // third constructor
bool bool_arr[] = { true, true, false }; // fourth constructor - size is defined automatically from the element count
for (int i = 0; i < arr.length(); i++) { // .length()
    // ...
}
const vec3 v[1] = {vec3(1.0, 1.0, 1.0)}; // global array

// struct (c-like)
struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
};
    
PointLight light = PointLight(vec3(0.0), vec3(1.0, 0.0, 0.0), 0.5);    

// always use epsilon for float comparison
const float EPSILON = 0.0001;
if (value >= 0.3 - EPSILON && value <= 0.3 + EPSILON) {
    // ...
}

// discard
discard // fragment & light shader can discard the color writing.
    
// inout: pass by reference    
// by default function params are only "in" for reading, but we can use inout to make it a reference.
void sum2(int a, int b, inout int result) {
    result = a + b;
}

//// varying: pass value from vertex to fragment & light shader.
// can only be assigned in vertex and fragment shader (not even in self-defined functions!)
varying float var_arr[3];
void vertex() {
    var_arr[0] = 1.0;
    var_arr[1] = 0.0;
}
void fragment() {
    ALBEDO = vec3(var_arr[0], var_arr[1], var_arr[2]); // red color
}

// varying values are usually interpolated from vertex to fragment shader, use `flat` or `smooth`(default) to change interpolation behaviour
varying flat vec3 color; // will use nearest color


// global uniforms: configured from project settings -> shader globals
global uniform vec4 mycolor;

// built-in math funcs
radians(); degrees();
sin(); cos(); tan(); ...
pow(); exp(); exp2(); log(); log2(); sqrt();
abs(); sign(); floor(); round(); trunc(); ceil(); mod(); min(); max();
isnan(); isinf(); 
length(); distance(); dot(); cross(); normalize(); reflect(); inverse();

// important shape functions
// most float can also be vec type
mix(float a, float b, float c); // linear interpolate, a * c + b * (1 - c)
step(float a, float b); // b < a ? 0.0 : 1.0
smoothstep(float a, float b, float c); // hermite interpolate (smooth), always return 0.0 if c < a, and 1.0 if c > b!
```

### Practices

```glsl
// polar coordinates
void fragment() {
	vec2 st = UV; // we usually call texture coordinate st other than uv
	st -= vec2(0.5); // rescale [0, 1] to [-0.5, 0.5]
	float r = length(st) * 2.0; // radius, rescale to [0, 1]
	float a = atan(st.y, st.x) + TIME; // angle, [-0.5PI, 0.5PI]
	float f = smoothstep(-0.5, 0.5, cos(a * 13.0)); // create a blurry edge using smoothstep, 13 controls frequency
	COLOR = vec4(vec3(f), 1.0);
}

// rotate 2d
mat2 rotate2d(float _angle){
    return mat2(vec2(cos(_angle),-sin(_angle)),
                vec2(sin(_angle),cos(_angle)));
}

// white noise using fract and sin
// we want pseudo-randomness, ie, given a position (st), the value is always fixed.
// these magic numbers control the pattern
float random1d(float x) {
    return fract(sin(x) * 43758.5453123);
}
float random2d(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.1233))) * 43758.5453123);
}

// grid-like randomness using fract
void fragment() {
	vec2 st = UV;
	st *= 10.0; // 10x10 grid
	vec2 ipos = floor(st); // we are still using float after floor!
	vec2 fpos = fract(st);
	//COLOR = vec4(fpos, 0.0, 1.0); // show grid
	COLOR = vec4(vec3(random2d(ipos)), 1.0);
}

// perlin noise (smoothed grid-like randomness)
float perlin1d(float x) {
    float i = floor(x);
    float f = fract(x);
	return mix(random1d(i), random1d(i + 1.0), smoothstep(0.0, 1.0, f));
}

float perlin2d(in vec2 st) { // st should be UV * grid_frequency
	vec2 i = floor(st);
	vec2 f = fract(st);
	float a = random2d(i);
    float b = random2d(i + vec2(1.0, 0.0));
    float c = random2d(i + vec2(0.0, 1.0));
    float d = random2d(i + vec2(1.0, 1.0));
	vec2 u = f * f * (3.0 - 2.0 * f); // 2d smoothstep!
	return a * (1.0 - u.x) * (1.0 - u.y) + b * u.x * (1.0 - u.y) + c * (1.0 - u.x) * u.y + d * u.x * u.y;
}

// simplex noise: more efficient
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289v2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
float snoise(vec2 v) {
    // Precompute values for skewed triangular grid
    const vec4 C = vec4(0.211324865405187,
                        // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,
                        // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,
                        // -1.0 + 2.0 * C.x
                        0.024390243902439);
                        // 1.0 / 41.0
    // First corner (x0)
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    // Other two corners (x1, x2)
    vec2 i1 = vec2(0.0);
    i1 = (x0.x > x0.y)? vec2(1.0, 0.0):vec2(0.0, 1.0);
    vec2 x1 = x0.xy + C.xx - i1;
    vec2 x2 = x0.xy + C.zz;
    // Do some permutations to avoid
    // truncation effects in permutation
    i = mod289v2(i);
    vec3 p = permute(permute( i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);

    m = m*m ;
    m = m*m ;

    // Gradients:
    //  41 pts uniformly over a line, mapped onto a diamond
    //  The ring size 17*17 = 289 is close to a multiple
    //      of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt(a0*a0 + h*h);
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0+h*h);

    // Compute final noise value at P
    vec3 g = vec3(0.0);
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * vec2(x1.x,x2.x) + h.yz * vec2(x1.y,x2.y);
    return 130.0 * dot(m, g);
}

// Fractual Brownian Motion noise (sharper than perlin & simplex, very useful for natural noise.)
float random (in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,8.233)))* 368.23);
}

float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (in vec2 st, int octaves) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    // Loop of octaves
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

void fragment() {
	vec2 st = UV * 5.0;
	COLOR = vec4(vec3(fbm(st, 6)), 1.0);
}

```



### Example Shaders

See here: https://godotshaders.com/shader-tag/2d/

Random zebra line: (example of Perlin noise)

```glsl
shader_type canvas_item;


float random (in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 458.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( random( i + vec2(0.0,0.0) ),
                     random( i + vec2(1.0,0.0) ), u.x),
                mix( random( i + vec2(0.0,1.0) ),
                     random( i + vec2(1.0,1.0) ), u.x), u.y);
}

mat2 rotate2d(float angle){
    return mat2(vec2(cos(angle),-sin(angle)),
                vec2(sin(angle),cos(angle)));
}

float lines(in vec2 pos, float b){
    float scale = 10.0;
    pos *= scale;
    return smoothstep(0.0, .5+b*.5, abs((sin(pos.x*3.1415)+b*2.0))*.5);
}

void fragment() {
	vec2 st = UV * 5.0;
	st = rotate2d(TIME + noise(st)) * st;
	COLOR = vec4(vec3(lines(st, 0.5)), 1.0);
}
```

Circle with waving border:

```glsl
shader_type canvas_item;

vec2 random2(vec2 st){
    st = vec2( dot(st,vec2(127.1,311.7)), dot(st,vec2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(st)*43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

mat2 rotate2d(float _angle){
    return mat2(vec2(cos(_angle),-sin(_angle)),
                vec2(sin(_angle),cos(_angle)));
}

float shape(vec2 st, float radius) {
	st = vec2(0.5)-st;
    float r = length(st)*2.0;
    float a = atan(st.y,st.x);
    float m = abs(mod(a+TIME*2.,3.14*2.)-3.14)/3.6;
    float f = radius;
    m += noise(st+TIME*0.1)*.5;
    //a *= 1.+abs(atan(TIME*0.2))*.1;
    //a *= 1.+noise(st+TIME*0.1)*0.1;
    f += sin(a*50.)*noise(st+TIME*.2)*.1;
    f += (sin(a*20.)*.1*pow(m,2.));
    return 1.-smoothstep(f,f+0.007,r);
}

float shapeBorder(vec2 st, float radius, float width) {
    return shape(st,radius) - shape(st,radius-width);
}

void fragment() {
	vec2 st = UV;
	COLOR = vec4(vec3(shape(st, 0.8)), 1.0);
	//COLOR = vec4(vec3(shapeBorder(st, 0.8, 0.02)), 1.0);
}
```

Rain:

```glsl
shader_type canvas_item;

uniform vec3 color: source_color = vec3(0.5);
uniform float speed: hint_range(0.01, 10.0, 0.01) = 0.1;
uniform float density: hint_range(1.0, 500.0, 1.0) = 100.0;
uniform float compression: hint_range(0.1, 1.0, 0.01) = 0.2;
uniform float trail_size: hint_range(5.0, 100.0, 0.1) = 50.0;
uniform float brightness: hint_range(0.1, 10.0, 0.1) = 5.0;

void fragment() {
	vec2 uv = -UV;
	float time = TIME * speed;
	uv.x *= density;
	vec2 duv = vec2(floor(uv.x), uv.y) * compression;
	float offset = sin(duv.x);
	float fall = cos(duv.x * 30.0);
	float trail = mix(100.0, trail_size, fall);
	float drop = fract(duv.y + time * fall + offset) * trail;
	drop = 1.0 / drop;
	drop = smoothstep(0.0, 1.0, drop * drop);
	drop = sin(drop * PI) * fall * brightness;
	float shape = sin(fract(uv.x) * PI);
	drop *= shape * shape;
	COLOR = vec4(color * drop, 1.0);
}
```

Metal-like Highlight:

```glsl
shader_type canvas_item;
render_mode blend_premul_alpha;

uniform float Line_Smoothness : hint_range(0, 0.1) = 0.045;
uniform float Line_Width : hint_range(0, 0.2) = 0.09;
uniform float Brightness = 3.0;
uniform float Rotation_deg : hint_range(-90, 90) = 30;
uniform float Distortion : hint_range(1, 2) = 1.8;
uniform float Speed = 0.7;
uniform float Position : hint_range(0, 1) = 0;
uniform float Position_Min = 0.25;
uniform float Position_Max = 0.5;
uniform float Alpha : hint_range(0, 1) = 1;

vec2 rotate_uv(vec2 uv, vec2 center, float rotation, bool use_degrees){
    float _angle = rotation;
    if(use_degrees){
        _angle = rotation * (3.1415926/180.0);
    }
    mat2 _rotation = mat2(
        vec2(cos(_angle), -sin(_angle)),
        vec2(sin(_angle), cos(_angle))
    );
    vec2 _delta = uv - center;
    _delta = _rotation * _delta;
    return _delta + center;
}

void fragment() {
	
	vec2 center_uv = UV - vec2(0.5, 0.5);
	float gradient_to_edge = max(abs(center_uv.x), abs(center_uv.y));
	gradient_to_edge = gradient_to_edge * Distortion;
	gradient_to_edge = 1.0 - gradient_to_edge;
	vec2 rotaded_uv = rotate_uv(UV, vec2(0.5, 0.5), Rotation_deg, true);
	
	float remapped_position;
	{
		float output_range = Position_Max - Position_Min;
		remapped_position = Position_Min + output_range * Position;
	}
	
	float remapped_time = TIME * Speed + remapped_position;
	remapped_time = fract(remapped_time);
	{
		float output_range = 2.0 - (-2.0);
		remapped_time = -2.0 + output_range * remapped_time;
	}
	
	vec2 offset_uv = vec2(rotaded_uv.xy) + vec2(remapped_time, 0.0);
	float line = vec3(offset_uv, 0.0).x;
	line = abs(line);
	line = gradient_to_edge * line;
	line = sqrt(line);
	
	float line_smoothness = clamp(Line_Smoothness, 0.001, 1.0);
	float offset_plus = Line_Width + line_smoothness;
	float offset_minus = Line_Width - line_smoothness;
	
	float remapped_line;
	{
		float input_range = offset_minus - offset_plus;
		remapped_line = (line - offset_plus) / input_range;
	}
	remapped_line = remapped_line * Brightness;
	remapped_line = min(remapped_line, Alpha);
	COLOR.rgb = vec3(COLOR.xyz) * vec3(remapped_line);
	COLOR.a = remapped_line;
}

```

Sway with wind:

```glsl
// original wind shader from https://github.com/Maujoe/godot-simple-wind-shader-2d/tree/master/assets/maujoe.simple_wind_shader_2d
// original script modified by HungryProton so that the assets are moving differently : https://pastebin.com/VL3AfV8D
//
// speed - The speed of the wind movement.
// minStrength - The minimal strength of the wind movement.
// maxStrength - The maximal strength of the wind movement.
// strengthScale - Scalefactor for the wind strength.
// interval - The time between minimal and maximal strength changes.
// detail - The detail (number of waves) of the wind movement.
// distortion - The strength of geometry distortion.
// heightOffset - The height where the wind begins to move. By default 0.0.

shader_type canvas_item;
render_mode blend_mix;

// Wind settings.
uniform float speed = 1.0;
uniform float minStrength : hint_range(0.0, 1.0) = 0.01;
uniform float maxStrength : hint_range(0.0, 1.0) = 0.5;
uniform float strengthScale = 100.0;
uniform float interval = 3.5;
uniform float detail = 1.0;
uniform float distortion : hint_range(0.0, 1.0);
uniform float heightOffset : hint_range(0.0, 1.0);

// With the offset value, you can if you want different moves for each asset. Just put a random value (1, 2, 3) in the editor. Don't forget to mark the material as unique if you use this
uniform float offset = 0; 


float getWind(vec2 vertex, vec2 uv, float time){
    float diff = pow(maxStrength - minStrength, 2.0);
    float strength = clamp(minStrength + diff + sin(time / interval) * diff, minStrength, maxStrength) * strengthScale;
    float wind = (sin(time) + cos(time * detail)) * strength * max(0.0, (1.0-uv.y) - heightOffset);
    
    return wind; 
}

void vertex() {
    vec4 pos = MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0);
    float time = TIME * speed + offset;
    //float time = TIME * speed + pos.x * pos.y  ; not working when moving...
    VERTEX.x += getWind(VERTEX.xy, UV, time);
}
```

Fake Camera Perepective:

```glsl
shader_type canvas_item;

// Camera FOV
uniform float fov : hint_range(1, 179) = 90;
uniform bool cull_back = true;
uniform float y_rot : hint_range(-180, 180) = 0.0;
uniform float x_rot : hint_range(-180, 180) = 0.0;
// At 0, the image retains its size when unrotated.
// At 1, the image is resized so that it can do a full
// rotation without clipping inside its rect.
uniform float inset : hint_range(0, 1) = 0.0;
// Consider changing this to a uniform and changing it from code

varying flat vec2 o;
varying vec3 p;

// Creates rotation matrix
void vertex(){
	float sin_b = sin(y_rot / 180.0 * PI);
	float cos_b = cos(y_rot / 180.0 * PI);
	float sin_c = sin(x_rot / 180.0 * PI);
	float cos_c = cos(x_rot / 180.0 * PI);
	
	mat3 inv_rot_mat;
	inv_rot_mat[0][0] = cos_b;
	inv_rot_mat[0][1] = 0.0;
	inv_rot_mat[0][2] = -sin_b;
	
	inv_rot_mat[1][0] = sin_b * sin_c;
	inv_rot_mat[1][1] = cos_c;
	inv_rot_mat[1][2] = cos_b * sin_c;
	
	inv_rot_mat[2][0] = sin_b * cos_c;
	inv_rot_mat[2][1] = -sin_c;
	inv_rot_mat[2][2] = cos_b * cos_c;
	
	
	float t = tan(fov / 360.0 * PI);
	p = inv_rot_mat * vec3((UV - 0.5), 0.5 / t);
	float v = (0.5 / t) + 0.5;
	p.xy *= v * inv_rot_mat[2].z;
	o = v * inv_rot_mat[2].xy;

	VERTEX += (UV - 0.5) / TEXTURE_PIXEL_SIZE * t * (1.0 - inset);
}

void fragment(){
	if (cull_back && p.z <= 0.0) discard;
	vec2 uv = (p.xy / p.z).xy - o;
    COLOR = texture(TEXTURE, uv + 0.5);
	COLOR.a *= step(max(abs(uv.x), abs(uv.y)), 0.5);
}
```

Dissolve:

Need to create a NoiseTexture2D with Simplex Smooth noise.

```glsl
shader_type canvas_item;

uniform sampler2D dissolve_texture : source_color;
uniform float dissolve_value : hint_range(0,1) = 0.5;
uniform float burn_size: hint_range(0.0, 1.0, 0.01) = 0.04;
uniform vec4 burn_color: source_color;

void fragment(){
    vec4 main_texture = texture(TEXTURE, UV);
    vec4 noise_texture = texture(dissolve_texture, UV);
	
	// This is needed to avoid keeping a small burn_color dot with dissolve being 0 or 1
	// is there another way to do it?
	float burn_size_step = burn_size * step(0.001, dissolve_value) * step(dissolve_value, 0.999);
	float threshold = smoothstep(noise_texture.x-burn_size_step, noise_texture.x, dissolve_value);
	float border = smoothstep(noise_texture.x, noise_texture.x + burn_size_step, dissolve_value);
	
	COLOR.a *= threshold;
	COLOR.rgb = mix(burn_color.rgb, main_texture.rgb, border);
}
```

Lightning:

```glsl
shader_type canvas_item;

uniform vec3 effect_color: source_color = vec3(0.2, 0.3, 0.8);
uniform float speed = 0.5;

// fbm params
uniform int octave_count: hint_range(1, 20) = 10;
uniform float amp_start = 0.5;
uniform float amp_coeff = 0.5;
uniform float freq_coeff = 2.0;


float hash12(vec2 x) {
    return fract(cos(mod(dot(x, vec2(13.9898, 8.141)), 3.14)) * 43758.5453);
}

vec2 hash22(vec2 uv) {
    uv = vec2(dot(uv, vec2(127.1,311.7)),
              dot(uv, vec2(269.5,183.3)));
    return 2.0 * fract(sin(uv) * 43758.5453123) - 1.0;
}

float noise(vec2 uv) {
    vec2 iuv = floor(uv);
    vec2 fuv = fract(uv);
    vec2 blur = smoothstep(0.0, 1.0, fuv);
    return mix(mix(dot(hash22(iuv + vec2(0.0,0.0)), fuv - vec2(0.0,0.0)),
                   dot(hash22(iuv + vec2(1.0,0.0)), fuv - vec2(1.0,0.0)), blur.x),
               mix(dot(hash22(iuv + vec2(0.0,1.0)), fuv - vec2(0.0,1.0)),
                   dot(hash22(iuv + vec2(1.0,1.0)), fuv - vec2(1.0,1.0)), blur.x), blur.y) + 0.5;
}

float fbm(vec2 uv) {
    float value = 0.0;
    float amplitude = amp_start;
    for (int i = 0; i < octave_count; i++) {
        value += amplitude * noise(uv);
        uv *= freq_coeff;
        amplitude *= amp_coeff;
    }
    return value;
}

void fragment() {
    vec2 uv = 2.0 * UV - 1.0;
    uv += 2.0 * fbm(uv + TIME * speed) - 1.0;
    float dist = abs(uv.x);
    vec3 color = effect_color * mix(0.0, 0.05, hash12(vec2(TIME))) / dist;
    COLOR = vec4(color, 1.0);
}
```

