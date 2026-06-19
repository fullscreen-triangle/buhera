# Buhera OS — Web REPL

Use the Buhera OS from a browser instead of a terminal. Same kernel,
same vaHera, same shortcuts; rendered in a black-on-grey terminal-style
page that talks to a local server process over plain HTTP.

This is **two pieces** running side by side:

1. **`buhera-server`** — a small Rust process that holds the kernel +
   embedder and exposes them as a JSON API on `localhost:5599`.
2. **`long-grass`** — the Next.js webtool that includes a `/os` page
   rendering a terminal-style REPL backed by the server.

Both run on your machine. Nothing leaves the laptop. The web page is
served by the Next.js dev server (or any static export); the kernel
runs in `buhera-server` and the browser talks to it over `fetch`.

---

## 1. Prerequisites

In addition to the [REPL prerequisites](./repl-use.md#1-prerequisites)
(Rust toolchain), you need:

- **Node.js 18 or newer**. Check with `node --version` and
  `npm --version`. If missing, install from
  [nodejs.org](https://nodejs.org/) or via a version manager
  (`nvm`, `fnm`, or `winget install OpenJS.NodeJS`).

---

## 2. Start the server

From the `buhera-os/` workspace:

```sh
# Semantic mode (downloads ~133 MB on first run).
cargo run --release -p buhera-server

# Lexical mode (no model, no download, builds in seconds).
cargo run --release -p buhera-server -- --lexical
```

You'll see something like:

```
(loading semantic embedder; first run downloads ~133 MB)
buhera-server  v0.1.0  bound 127.0.0.1:5599  embedder=bge-small-en-v1.5  depth=12
endpoints:
  GET  /info
  GET  /stats
  GET  /list
  GET  /find?q=...&k=N&rerank=true
  POST /store           {"name":..., "text":...}
  POST /vahera          {"source":..., "rerank":bool}
  DELETE /memory        clear the kernel
```

Leave the server running. It holds the kernel for the lifetime of the
process; restart to start fresh.

### Server command-line flags

| Flag                | Default              | Effect                                                                  |
| ------------------- | -------------------- | ----------------------------------------------------------------------- |
| `--bind <addr>`     | `127.0.0.1:5599`     | Listen address. Use `0.0.0.0:5599` to allow other devices on your LAN.  |
| `--depth <N>`       | `12`                 | Ternary-address depth used by CMM.                                      |
| `--lexical`         | off                  | Use the hash-bag embedder instead of the semantic model.                |
| `--permissive-cors` | on                   | Allow CORS from any origin (so the webtool works no matter the port).   |

---

## 3. Start the webtool

In a second terminal (the server has to keep running):

```sh
cd long-grass
npm install     # first time only
npm run dev
```

You'll see Next.js print something like:

```
- ready started server on 0.0.0.0:3000, url: http://localhost:3000
```

Open <http://localhost:3000/os> in your browser. You should see:

```
buhera-os web repl     bge-small-en-v1.5 · depth 12 · 0 objects     server: http://localhost:5599
```

If the top-right shows `offline` and the buffer prints `could not
reach http://localhost:5599`, the server isn't running or is on a
different port. Either start the server (step 2) or edit the address
in the `server:` input box at the top right.

---

## 4. Use it

Type into the prompt at the bottom of the page. Same shortcuts as the
desktop REPL:

```
store apples = "I want to buy apples and bread"
store work   = "the deadline for the proposal is Friday"
store gym    = "go to the gym on Saturday morning"

find "shopping"
find "deadline"
find "workout"

list
stats
```

Arrow-up / arrow-down navigate the command history. Enter submits.

### Meta-commands in the web REPL

| Command   | Effect                                  |
| --------- | --------------------------------------- |
| `:help`   | Show shortcuts and statement reference. |
| `:tour`   | Load a short guided demo.               |
| `:info`   | Show server info: embedder, depth, count. |
| `:clear`  | Reset the kernel (calls `DELETE /memory`). |

`:quit` doesn't apply (just close the browser tab). To restart the
kernel, use `:clear`, or restart the `buhera-server` process.

---

## 5. The HTTP API

You can also talk to the server directly with `curl`, `httpie`, or any
HTTP client — the webtool is just one consumer of this API.

### `GET /info`

```json
{
  "version": "0.1.0",
  "embedder": "bge-small-en-v1.5",
  "depth": 12,
  "objects": 0
}
```

### `POST /store`

```sh
curl -X POST http://localhost:5599/store \
     -H 'Content-Type: application/json' \
     -d '{"name":"hello","text":"hello world from curl"}'
```

Returns the allocated object including its categorical address.

### `GET /find?q=<text>&k=<n>&rerank=<bool>`

```sh
curl 'http://localhost:5599/find?q=greeting&k=3'
```

Returns a JSON array of hits.

### `POST /vahera`

Send any vaHera script:

```sh
curl -X POST http://localhost:5599/vahera \
     -H 'Content-Type: application/json' \
     -d '{"source":"memory store \"note\" = \"any text\"\nkernel stats","rerank":true}'
```

Returns the interpreter trace plus the structured results.

### `GET /stats`, `GET /list`, `DELETE /memory`

Self-explanatory. Stats matches the REPL's `kernel stats`; list returns
every stored object; `DELETE /memory` resets the kernel.

---

## 6. Production deployment notes

The server is intended for local use. If you want to expose it on a
LAN or beyond, change `--bind` from `127.0.0.1` to `0.0.0.0`, but
**add authentication first** — the API has no built-in auth, and
anyone reachable can store, search, and clear your kernel.

For a hosted demo at the long-grass site, you'd either:

- Bundle the model into a single binary and run `buhera-server`
  behind a reverse proxy.
- Switch the webtool's `server:` address input to point at the hosted
  endpoint.

Both are out of scope for v0.2.x; the design assumes desktop-local use.

---

## 7. Troubleshooting

### "could not reach http://localhost:5599"

The server isn't running, or it's bound to a different port. Check
the server window for its bound address; edit the `server:` field in
the webtool to match.

### "syntax error at line 1: unknown vaHera: ..."

The line starts with a vaHera keyword (`memory`, `kernel`, etc.) but
the rest doesn't match the grammar. Try `:help` for the menu, or use
the friendly `store`/`find` shortcuts which translate for you.

### `npm install` fails with a node version error

The webtool needs Node 18+. Upgrade Node or use a version manager.

### Tests pass but the page doesn't show search results

Check the browser console (`F12` → Console). The most common cause is
a CORS error on a non-default port; rerun the server with
`--permissive-cors` (default on) or restrict the front-end to
`localhost:3000`.

### Restart vs. clear

- `:clear` in the webtool wipes the kernel without restarting the server.
  Embedding model stays loaded; tons faster.
- Stopping/restarting `buhera-server` re-loads the model and starts
  the kernel fresh. Use this if you want to switch from semantic to
  lexical or change the depth.

---

## 8. What's running where

```
                ┌────────────────────────┐
                │   browser (localhost)  │
                │   long-grass /os page  │
                └───────────┬────────────┘
                            │  HTTP (fetch)
                            │
                            ▼
                ┌────────────────────────┐
                │     buhera-server      │
                │   (port 5599 by def.)  │
                │   ┌──────────────┐     │
                │   │   Kernel     │     │
                │   │   Embedder   │     │
                │   └──────────────┘     │
                └────────────────────────┘
```

The browser holds *zero* OS state. The page is stateless React
rendering of whatever the server returns; refresh and the prompt comes
back empty, but the kernel keeps everything you put in it.
