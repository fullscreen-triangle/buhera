# Buhera OS — Web Demo

A no-install, in-browser demo of the Buhera OS. Open a page, type, the
OS responds. Same five subsystems (CMM, PSS, DIC, PVE, TEM), same
vaHera language, same categorical addressing — implemented in
JavaScript so it runs in any modern browser without a server.

This is the "before you commit" version. When you're ready for the
real OS with the semantic embedder and the full Rust kernel, see
[`repl-use.md`](./repl-use.md).

---

## 1. Run it locally

```sh
cd long-grass
npm install   # first time only
npm run dev
```

Open <http://localhost:3000>. The Buhera terminal is the homepage.
You'll see a welcome banner and a prompt.

---

## 2. Try it

The terminal accepts three flavours of input. You don't have to choose
— just type what you mean.

### Plain text — store and search

```
store note   = "remember to send the proposal to Sarah on Friday"
store milk   = "buy milk eggs and bread on the way home"
store gym    = "go to the gym on Saturday morning"
find "shopping"
find "writing"
list
stats
```

`store` and `find` are the two everyday verbs. Everything else is
optional.

### Shortcuts

| What you type             | What runs                                  |
| ------------------------- | ------------------------------------------ |
| `store <name> = "<text>"` | `memory store "<name>" = "<text>"`         |
| `find "<text>" [k=N]`     | `memory find nearest "<text>" k=N`         |
| `list`                    | `memory list`                              |
| `dump <name>`             | `memory dump <name>`                       |
| `sort`                    | `demon sort`                               |
| `stats`                   | `kernel stats`                             |
| `trace`                   | `kernel trace`                             |
| `verify`                  | `controller verify`                        |
| `procs` or `ps`           | `process list`                             |

A bare line with no keyword is treated as a search query.

### Raw vaHera

Any of the 15 statements works directly:

```
describe sarah with "Sarah Chen, product manager"
spawn lookup_sarah from sarah
navigate to penultimate
complete trajectory
```

You can paste a full multi-line vaHera script too.

### Meta commands

| Command      | Effect                                                                        |
| ------------ | ----------------------------------------------------------------------------- |
| `:tour`      | Load five sample notes and run three searches against them.                   |
| `:proteins`  | Load the hardcoded biology demo and enable natural-language questions.        |
| `:clear`     | Reset the kernel to empty.                                                    |
| `:help`      | Show every vaHera statement.                                                  |

---

## 3. The proteins demo

The original Buhera homepage shipped with a biology database and a
pattern-matching NL → vaHera translator. That still works — type
`:proteins` to load it. From then on, natural-language questions like

```
tell me about TP53
function of BRCA1
diseases linked to MYC
compare BRCA1 and BRCA2
```

are routed through the translator and rendered as protein records.

`:clear` removes the loaded proteins and goes back to blank.

---

## 4. What it actually does

When you `store note = "..."`:

1. The text gets tokenised, each token hashed (FNV-1a), and the hashes
   are summed to give a three-axis coordinate `(S_k, S_t, S_e)`.
2. A 12-character ternary address is computed from the coordinate.
3. CMM stores the payload at that address, classifies the tier by
   `||S||`, and records an event.
4. TEM (Triple Equivalence Monitor) samples the coordinate.

When you `find "..."`:

1. The query embeds to a coordinate.
2. CMM does a proximity scan (Fisher metric in S-space).
3. DIC performs surgical retrieval of the top-k candidates.
4. A token-overlap re-ranker rescues hits that literally share words
   with the query — useful when the lexical embedding alone is too
   ambiguous to pick the right note.
5. Results render in the terminal.

Everything is deterministic. Same input, same output, same address,
on any browser.

---

## 5. What it does not have

- **No semantic model.** The webtool is lexical-only (token-bag with
  light biases). It catches lexical overlap (`"shopping"` → `"buy
  groceries"`) thanks to the re-ranker, but it won't infer synonyms
  outside the training distribution.
- **No persistence.** Refreshing the page clears the kernel. State
  lives only in the browser tab.
- **No multi-process kernel.** The JS kernel runs in the main thread;
  PSS is a flat ready queue, not a scheduler.

If you need a semantic embedder (BGE-Small running locally), full
process scheduling, or persistent storage, switch to the desktop OS
([`repl-use.md`](./repl-use.md)). The webtool exists to give you a
working feel for the categorical-addressing model before you commit
to the install.

---

## 6. Deploy

```sh
cd long-grass
npm run build
npm start
```

Or export static and host anywhere:

```sh
npx next build
npx next export
```

The `out/` directory is plain HTML/JS — drop it on Vercel, Netlify,
GitHub Pages, or any static host. The kernel runs entirely in the
visitor's browser; you have no backend to maintain.
