# Scanned Objects MuJoCo Models

This repository contains [MuJoCo](https://mujoco.org/) MJCF models for all 1030 household objects in Google's [Scanned Objects Dataset](https://ai.googleblog.com/2022/06/scanned-objects-by-google-research.html). The XML files were automatically generated using [obj2mjcf](https://github.com/kevinzakka/obj2mjcf). A sample of the models is shown below:

<p float="left">
  <img src="assets/clock.png" width="200"/>
  <img src="assets/basket.png" width="200"/>
  <img src="assets/plate.png" width="200"/>
  <img src="assets/mouse.png" width="200"/>
  <img src="assets/stack.png" width="200"/>
  <img src="assets/rack.png" width="200"/>
  <img src="assets/shoe.png" width="200"/>
  <img src="assets/hammer.png" width="200"/>
</p>

## Structure

In each object directory, you'll find the following:

```bash
model.obj
model_collision_*.obj  # 32 collision submeshes.
texture.png
model.xml
```

The model definition lives in `model.xml`. To visualize it, you can drag-and-drop it in the simulate binary. You can examine the visual and collision meshes by cycling between groups 2 and 3. The collision mesh is a convex approximation that has been generated with [V-HACD](https://github.com/kmammou/v-hacd).

If there are any problems loading a particular model in MuJoCo, please raise an issue on the [GitHub repository](https://github.com/kevinzakka/mujoco_scanned_objects).

## License

The 3D assets (obj and png) are licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). The MJCF (XML) files are licensed under the [MIT License](LICENSE).

## Citation

If you use these models in your research, consider citing the original paper:

```bibtex
@misc{downs2022scannedobjects,
    url = {https://arxiv.org/abs/2204.11918},
    author = {Downs, Laura and Francis, Anthony and Koenig, Nate and Kinman, Brandon and Hickman, Ryan and Reymann, Krista and McHugh, Thomas B. and Vanhoucke, Vincent},
    keywords = {Robotics (cs.RO), Graphics (cs.GR), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items},
    publisher = {arXiv},
    year = {2022},
}
```

If you're feeling generous, you can also cite this repository:

```bibtex
@software{zakka2022scannedobjectsmujoco,
    author = {Zakka, Kevin},
    month = {7},
    title = {{Scanned Objects MuJoCo Models}},
    url = {https://github.com/kevinzakka/mujoco_scanned_objects},
    version = {0.0.1},
    year = {2022}
}
```

_Disclaimer: This is not affiliated with Scanned Objects or Google Research._

## In this repository (locomanipulation, 2026-11-03)

Only `catalog.json`, `categories.json`, `LICENSE`, `README.md` and `VERSION` are tracked; `models/` (2 GB) is installed by
`bash scripts/install_gso.sh` (a shallow clone of kevinzakka/mujoco_scanned_objects, or `--from <dir>`) and is in
`.gitignore`. The scenes `logistics_room_1030_s<seed>` (`configs/scenes.yaml`) draw ten objects per instance from
`catalog.json` and copy those objects' meshes into the scene directory; see `docs/rsi_logistic_room_1030.md`.

`categories.json` (`scripts/gso_categories.py`): every model's category as the dataset labels it on Gazebo Fuel
(Shoe 254, Consumer Goods 248, Toys 150, Bottles and Cans and Cups 53, Bag 28, Media Cases 21, Action Figures 17,
Board Games 17, Legos 10, Headphones / Keyboard / Mouse 4 each, Stuffed Toys 3, Hat 2, Camera 1, Car Seat 1; 216
unlabelled); this conversion ships no labels. Only objects of the categories in `--use-categories` can be drawn; the
catalog marks every other object `none`. The default (user 2026-11-05, `gso_objects.USED_CATEGORIES`) is every label but
Shoe, Toys, Stuffed Toys, Mouse, Media Cases, Legos, Keyboard, Car Seat, Action Figures and Hat, plus the unlabelled
models (`(none)`); `--use-categories all` uses everything, `--exclude-categories A B` everything but those.

`catalog.json` (`scripts/gso_catalog.py`, `locomani.sim.gso_objects`): `version`, `source` (dataset, `VERSION`,
models dir, categories file), `thresholds` (the rule's parameters as run, `used_categories` among them), `not_used_categories`, `counts`
(`excluded` counts the objects of the categories not used, which are inside `none`), and per object under `objects`
keyed `gso_<4-digit index in sorted directory order>`:

| Field | Meaning |
|---|---|
| `name`, `phrase` | the directory name; a readable phrase from it (scan id dropped, lower case) |
| `category` | the dataset's label from `categories.json` (`null` when it has none) |
| `extent_m`, `bbox_min_m`, `bbox_max_m` | the visual mesh's axis-aligned box (the scans stand on z = 0, origin at the bottom centre) |
| `hull_volume_m3`, `n_hulls`, `n_vertices` | the V-HACD hulls' summed volume (the mass's volume) |
| `mass_kg` | `hull_volume_m3` x 300 kg/m^3, clamped to 0.05-1.5 kg (one fixed mass per object, user Q3) |
| `grasp_height_m` | the predicted single-hand grasp point above the floor: platform top 0.371 m + half the height (Q10) |
| `rest` | a 24 s drop test on a plane: `tilt_deg`, `slid_mm`, `dz_mm`, `stands` |
| `graspable`, `why` | `single` (narrow side fits the 9.5 cm Dex1 jaw, 3 cm or taller, grasp point 0.40 m or higher, stands), `bimanual` (wide side 0.20-0.45 m, 10 cm or taller, stands; untested, Q17), or `none` (also every object of a category not used: `why` says `category Shoe: not among the used categories`) |
| `split` | `train` or `held_out` (10 % by name hash; `--pool eval_unseen` instances draw the held-out tenth, Q7 / Q22) |

Attribution: the scans are Google Research's Scanned Objects dataset (Downs et al. 2022, CC-BY 4.0); the MJCF files
and hulls are Kevin Zakka's (MIT). Objects appear in rendered images and task prompts under their dataset names.
