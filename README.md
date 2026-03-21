# viser2blender

`viser2blender` converts a strict supported subset of `.viser` recordings into
`.blend` files through `bpy`.

It is packaged separately from `viser4d` because it requires `numpy<2` and
`bpy>5.0.0`.

Quick manual check:

```bash
cd ../viser2blender
uv run --python 3.11 viser2blender \
  tests/assets/blender_showcase.viser \
  /tmp/blender-showcase.blend \
  --overwrite
```
