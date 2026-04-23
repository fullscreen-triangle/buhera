# `/purpose` web surface

Client for the Rust Purpose CLI. Lives at `/purpose` in the long-grass
site; its API route lives at `/api/purpose`.

## Contract with the Rust side

Integration is exclusively via the `purpose` CLI subprocess (integration
contract §5, §9.4). The web tool never imports a Purpose Rust crate, and
the Rust side never calls the web tool.

```
POST /api/purpose
Content-Type: application/json
{
  "utterance": "Tell me about SOD1",
  "mode": "value" | "fragment"       // default: "value"
}

-> 200 { "ok": true, "value": <Value JSON>, "elapsed_ms": <number> }
-> 200 { "ok": true, "fragment": <vaHera JSON>, "elapsed_ms": <number> }
-> 4xx / 5xx { "ok": false, "error": "...", "stderr"?: "...", "elapsed_ms"?: <number> }
```

The backend spawns:

```
purpose query <utterance> --raw           # value mode
purpose query <utterance> --fragment      # fragment mode (compile only)
```

and expects a single JSON document on stdout.

## Binary resolution

The API route resolves the CLI in this order:

1. `$PURPOSE_CLI` (absolute path to the binary).
2. `../mechanistic-synthesis/implementation/target/release/purpose[.exe]`
3. `../mechanistic-synthesis/implementation/target/debug/purpose[.exe]`

If none exists the route returns 503 with a message pointing at the
build command. The site itself still builds and boots without the
binary — only the submit action fails.

## Safety

* Stdout is capped at 2 MiB and the subprocess is killed on overrun.
* No shell involved; `child_process.spawn` is used directly with
  `windowsHide: true`.
* Only `POST` with `{utterance: string, mode?: string}` is accepted.

## Running locally

```bash
# 1. Build the Rust binary
cd mechanistic-synthesis/implementation
cargo build --release

# 2. Start the web site
cd ../../long-grass
npm install
npm run dev

# 3. Open http://localhost:3000/purpose
```
