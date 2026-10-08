# The Landing Document

After you sign in, Buhera opens on a document you run, not a page somebody made for you. It holds prose and **cells**. A cell is a question, a search or a script; running it writes the result underneath and replaces the last one, so every visit can give a different answer. Every run is also kept in the **record**, which only grows.

**Time:** 10 minutes.

**Before you start:** on the hosted site, the document answers only its owner (see §6). Scripts and file search run only from long-grass on your own computer (`npm run dev` in `long-grass/`, then `http://localhost:3000`).

---

## 0. What the document does, and what it never does

| It does | It never does |
|---|---|
| search the web, fresh on every run | cache an old answer and show it as new |
| let Claude search, read, and answer with its sources | let the model run a script by itself |
| run your Python, bash, PowerShell or Node scripts on your PC | run a script from the hosted site on its server |
| keep every run in an append-only record | delete or rewrite the record |

## 1. The command line

The line at the top adds a cell to the end of the document and runs it. What you write decides the kind; the label to its left shows which:

| Write | Kind | What runs |
|---|---|---|
| any question | `ask` | Claude searches the web, the pages you have read and (on your PC) your files, then answers with links |
| `web <words>` | `web` | a search engine's results |
| `read <address>` | `read` | the page is read and kept in your library |
| `find <words>` | `find` | what you have read, searched with a verdict |
| `files <words>` | `files` | files under your Documents folder, by name and by content |
| `$ <command>` · `> <command>` | `bash` · `powershell` | a shell script |
| `py <code>` · `node <code>` | `python` · `node` | a script |

```
web DCAT-AP-PLUS LinkML
```

```text
web "DCAT-AP-PLUS LinkML" · 10 results · duckduckgo · 2026-10-08 11:03 UTC
- GitHub - nfdi-de/dcat-ap-plus: A domain-agnostic extension of the DCAT …
- DCAT-AP Plus Links to Use-case Specific Context (DCAT-AP+) — … a provenance layer for describing how a dataset was generated …
- dcat-ap-plus · PyPI — dcat-ap-plus 0.1.0rc4 …
```

## 2. Cells

Each cell has its kind (a menu: change it to rerun the same text another way), an info field, **▶ run** and, on hover, ↑ ↓ **clear** **delete**. Ctrl+Enter inside a cell runs it. Edit the text and run again: the output below is replaced.

Hover between two blocks for **+ text** (prose, in Markdown; click prose to edit it) and **+ cell**. The document saves itself a moment after each change, to `~/.buhera/notebook/today.md` on the machine the server runs on. `/?doc=lipids` opens another document, `lipids.md`.

## 3. Asking

```text
What does DCAT-AP+ add to DCAT-AP 3.0.1?
```

While it runs, the cell lists each search and page as it happens (`web "…"`, `read https://…`, `library "…"`) and the answer streams in under them. The answer ends with the model and everything it looked at, so you can check its sources. If the best answer is a script, it writes one, and **+ add the python as a cell** puts it in the document for you to read and run. The model never runs it.

The info field picks the model: `claude` or `hf`. Without one, Claude answers when the server has a Claude key, and a Hugging Face model otherwise. The Hugging Face model gets the same searches made for it in advance; it cannot search further by itself.

## 4. Scripts

```text
py import sys, datetime; print(sys.version.split()[0], datetime.datetime.now().isoformat(timespec='seconds'))
```

```text
3.14.3 2026-10-08T13:03:32

exit 0 · 0.3 s
```

Scripts run in `~/.buhera/notebook/work/`, so files they write stay together. Output streams in as it is printed. A script stops after 60 seconds unless the info field says otherwise (`timeout=600` at most). It gets a clean environment: the server's keys and mail passwords are not passed to it, so it cannot print them into the document.

## 5. The record

**record · N** at the top lists every run of this document, newest first, with its output. Clearing an output, deleting a cell, or running it again changes the document, never the record. The record is `~/.buhera/notebook/record.jsonl`, one line per run.

## 6. Setting it up

In `long-grass/.env.local`, then restart the server:

```text
ANTHROPIC_API_KEY=…                 # Claude; or, through a gateway:
ANTHROPIC_BASE_URL=…  ANTHROPIC_AUTH_TOKEN=…
HUGGINGFACE_API_KEY=…               # the fallback model (HF_NOTEBOOK_MODEL picks it)
BUHERA_OWNER=<your account id>      # on the hosted site: who may run the document
NOTEBOOK_SEARCH_ROOTS=…             # folders `files` searches (default: Documents)
```

`NOTEBOOK_MODEL` (default `claude-opus-5-5`) and `NOTEBOOK_EFFORT` (default `medium`) tune Claude. On the hosted site, a signed-in account that is not `BUHERA_OWNER` is refused. If no owner is set yet, the refusal names your account id, so you can set it.

Next: [The Blank Screen](./the-blank-screen), the other way of working — at `/surface`.
