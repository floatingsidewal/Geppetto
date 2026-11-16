---
name: geppetto
description: Turn pictures into 3D models (STL for printing) with the local Geppetto server. `/geppetto up` starts the web UI, `/geppetto process <file>` converts one image, `/geppetto process all [dir]` converts every image in a directory. Use when the user says "turn this into a 3D model", "make an STL of this", "image to 3D", "make this printable", or points at an image and asks for something to print.
argument-hint: up | process <file> [--overwrite] | process all [dir] [--overwrite]
---

# Geppetto — picture to 3D model

Everything goes through one script, `geppetto.py`, in the same directory as this
file. It needs only the Python standard library. Do not read the Geppetto repo,
its HOWTO, or probe Docker first — the script starts the server itself, waits
for the model, and reports clear errors.

```bash
python "${CLAUDE_SKILL_DIR}/geppetto.py" $ARGUMENTS
```

Use `python3` where `python` is not Python 3.

| Command | What it does |
|---|---|
| `up` | Starts the server if needed, waits for the model, opens the web UI at `/app` |
| `process <file> [<file>...]` | Writes `<name>.stl` beside each image |
| `process all [dir]` | The same for every image in `dir` (default: current directory, not recursive) |

Options for `process`:

- `--overwrite` — replace models that already exist. **Never add this unless the
  user asked for it.** Without it a single file is refused (exit 2, nothing
  generated) and a batch skips the images that already have a model.
- `--format stl|glb|obj|ply` — repeatable; default `stl`. GLB is the only format
  that carries the texture.
- `--resolution 512|1024|1536` — default 1024.

When the user gives an image without a subcommand ("make an STL of this"), run
`process` on it.

## Running it

- A cold start takes about 90 seconds and each image takes 2–5 minutes. Run the
  command with a 10-minute timeout; run `process all` in the background.
- The server is `http://localhost:8080` unless `GEPPETTO_URL` says otherwise.
- If the script fails, report its message verbatim. Do not retry a failed
  generation: it is expensive and a real failure repeats.

## Reporting

Give the path of each file written. Mention once that meshes are not
watertight, so a slicer repair pass may be needed before printing, and that STL
has no colour (offer `--format glb` if they seemed to want the texture).
