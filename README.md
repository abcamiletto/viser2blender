# viser2blender

Convert a strict supported subset of `.viser` recordings into `.blend` files.

Requirements:
- Python 3.13
- `bpy>=5.1.0`

Usage:

```bash
uvx viser2blender input.viser output.blend --overwrite
```

Optional flags:
- `--validate-only`
- `--emit-manifest manifest.json`
