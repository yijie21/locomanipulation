# Intel RealSense D435 — wrist camera body

`d435.obj` is the visible body for the wrist cameras (`scene_lib.py`,
`add_wrist_cameras`). It is **visual only**: no contact, no mass. Nothing in
the control path reads it.

## Where it came from

Intel's own ROS description package, which is the mesh everyone else's
simulation uses:

    https://github.com/IntelRealSense/realsense-ros
    realsense2_description/meshes/d435.dae   (ref ros2-master, fetched 2026-08-28)

Licence: **Apache-2.0** (the realsense-ros repository LICENSE covers it).
Copyright Intel Corporation.

## Why it is not the file they ship

The original is a 15.8 MB COLLADA file with **231,186 triangles** — a CAD
export, for a prop that is ~50 px across in the ego view. MuJoCo also does not
read COLLADA. So it is converted and decimated to 20,000 triangles / 1.3 MB,
which keeps the outside dimensions exact (89.9 x 25.0 x 25.1 mm measured before
and after) and keeps the four lens apertures and the vent grille legible.

Reproducing it needs only Blender, which is already on the box:

```bash
curl -sSLO https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-master/realsense2_description/meshes/d435.dae
python dae2obj.py d435.dae d435_raw.obj      # positions + triangles; the file
                                             # has one node and no transforms
blender -b --python dec.py -- d435_raw.obj d435.obj 20000
```

`dec.py` does, in order: weld (`remove_doubles`, 1e-5 — the DAE ships 18
separate parts and decimation cannot collapse across an unwelded seam),
**planar** dissolve at 5 deg (this is a CAD part: dissolving coplanar faces
keeps the silhouette exactly, where a collapse rounds the corners off),
triangulate, then collapse to the cap. Both scripts are short enough to
rewrite; they were one-offs and are not carried in the repo.

## Frame

Convenient as shipped, and `scene_lib` relies on it:

* **+z is the lens direction, and z = 0 is exactly the lens plane** (the body
  occupies z in [-0.025, 0])
* +y is up, x is the 90 mm width, centred

A MuJoCo camera looks along its own **-z**, so the geom is the camera's
orientation turned 180 deg about y, placed at the camera's own position. That
puts the lens plane on the camera origin and the whole body behind it — which
is why the wrist camera never sees its own housing.
