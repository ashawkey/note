

### PLY loader example

`index.html`

```html
...

<body>
    <div di="hook">
    </div>
</body>

<!-- always load the jsm at last (after the used element is created!) -->
<script type="module" src="static/js/renderer.js"> </script>

...
```


`renderer.js`

```js
import * as THREE from 'https://cdn.skypack.dev/three@0.132.0';
import { PLYLoader } from 'https://cdn.skypack.dev/three@0.132.0/examples/jsm/loaders/PLYLoader.js';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.0/examples/jsm/controls/OrbitControls.js';
import Stats from 'https://cdn.skypack.dev/three@0.132.0/examples/jsm/libs/stats.module.js';


let container, plyloader;

let camera, scene, renderer, controls;

init();

function init() {
	// append the canvas under the #hook element in html.
    container =  document.getElementById('hook');

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xffffff );
	
    // camera
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.set( 0, 0, 10 );
    camera.up.set( 0, 0, 1 ); // important! we want to rotate around z axis.

    // renderer
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( 0.8 * window.innerWidth, 0.8 * window.innerHeight );
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;

    container.appendChild( renderer.domElement );

    // controls
    controls = new OrbitControls( camera, renderer.domElement );
    controls.listenToKeyEvents( window ); // optional
    controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    // controls.minDistance = 100;
    // controls.maxDistance = 500;
    // controls.maxPolarAngle = Math.PI / 2;

    // PLY file Loader
    plyloader = new PLYLoader();

    // Lights
    scene.add( new THREE.HemisphereLight( 0xbdbdbd, 0x111122 ) );
    addShadowedLight( 10, 10, 10, 0x888888, 1.35 );

    // resize
    window.addEventListener( 'resize', onWindowResize );

}

function post_load_ply (type) {
    if (type === "points_colored") {
        return (geometry) => {
            const material = new THREE.PointsMaterial( { vertexColors: true, size: 0.01 } ); // enable vertex color from ply.
            const points = new THREE.Points( geometry, material );
            scene.add( points );
            // add fake ground
            geometry.computeBoundingBox();
            let bbox = geometry.boundingBox;
            add_plane(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y, bbox.min.z);
        }
    } else if (type === "points") {
        return (geometry) => {
            const material = new THREE.PointsMaterial( { color: 0x000000, size: 0.02 } ); // disable color from ply
            const points = new THREE.Points( geometry, material );
            scene.add( points );
        }
    } else if (type === "pred") {
        return (geometry) => {
            const material = new THREE.MeshPhongMaterial( { color: 0x888888, flatShading: true, side: THREE.DoubleSide } );
            //const material = new THREE.MeshStandardMaterial( { color: 0x888888, flatShading: true } );
            //const material = new THREE.MeshBasicMaterial( { color: 0x0000ff } );
            const mesh = new THREE.Mesh( geometry, material );
            scene.add( mesh );
        }
    }
}

function addShadowedLight( x, y, z, color, intensity ) {

    const directionalLight = new THREE.DirectionalLight( color, intensity );
    directionalLight.position.set( x, y, z );
    scene.add( directionalLight );

    directionalLight.castShadow = true;

    const d = 1;
    directionalLight.shadow.camera.left = - d;
    directionalLight.shadow.camera.right = d;
    directionalLight.shadow.camera.top = d;
    directionalLight.shadow.camera.bottom = - d;

    directionalLight.shadow.camera.near = 1;
    directionalLight.shadow.camera.far = 4;

    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;

    directionalLight.shadow.bias = - 0.001;

}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {

    requestAnimationFrame( animate );

    controls.update();
    renderer.render( scene, camera );

}

function load_input () {
    let scene_name = document.getElementById('scene_name').value;
    console.log("[INFO] load input for ", scene_name);
    // clear scene
    clear_scene()
    // load new input
    plyloader.load('static/RFS/input/' + scene_name + '_semantic_pred.ply', post_load_ply("points") );
}

function change_k () {
    let K = document.getElementById('select_k').value;
    console.log("[INFO] change K_projection ", K);
}

function run () {
    // disable run button first
    let button = document.getElementById("run");
    button.disabled = true;
    button.value = "运行中";

    let scene_name = document.getElementById('scene_name').value;
    let K_projection = document.getElementById('select_k').value;

    // call backend (fetch)
    fetch('/rfs/run/' + scene_name + '/' + K_projection).then((resp) => {
        return resp.json();
    }).then((res) => {
        // check error
        if (res['success']) {
            // load_pred
            clear_scene()
            plyloader.load('static/RFS/output/' + scene_name + '_semantic_pred.ply', post_load_ply("points_colored") );
            res['preds'].forEach((pred, idx) => {plyloader.load(pred, post_load_ply("pred") );});

            // reset button
            button.disabled = false;
            button.value = "运行";
        }
    });
}

function clear_scene () {
    for (let i = scene.children.length - 1; i >= 0; i--) {
        if(scene.children[i].type === "Mesh" || scene.children[i].type === "Points")
            scene.remove(scene.children[i]);
    }
}

function add_plane(h, w, z) {
    // Ground
    const plane = new THREE.Mesh(
        new THREE.PlaneGeometry( h, w ),
        new THREE.MeshPhongMaterial( { color: 0xaaaaaa, specular: 0x101010, side: THREE.DoubleSide } )
    );
    plane.position.z = z;
    plane.receiveShadow = true;
    scene.add( plane );
}


// bind to window
window.load_input = load_input;
window.change_k = change_k;
window.run = run;

load_input();
change_k();

animate();
```

