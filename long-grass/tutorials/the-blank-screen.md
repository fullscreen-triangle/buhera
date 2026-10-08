# The Blank Screen

The blank surface, at `/surface` (the **surface** link at the top of the [landing document](./the-landing-document)), is a black screen with a caret and nothing else. There are no windows, no menus and no app to choose first: you write what you want, and each thing you do becomes a page of its own. This tutorial shows how to write, how to go back, where everything else is, and how to put parts of several pages side by side.

**Time:** 10 minutes.

**The running example.** The next three tutorials carry one experiment through: a lipid series (PC 34:1) to run on the lab's LARA robot, with a colleague, Mara, organising the robot time. You will find what your mail and notes already say about it, plan it, and score its data on AppHub.

---

## 1. Writing

Click anywhere and type. Enter commits what you wrote; Shift+Enter starts a new line. What you write can be in any syntax:

- **A few words of a known verb** run straight away: `mail …`, `find …`, `plan …`, `apphub`, `chart …` (all below).
- **A script** in one of Buhera's languages runs as written. vaHera is the native one:

```
describe sample with "PC 34:1 lipid series"
```

- **Anything else** goes to the player: retrieval over your own notes plus your personal model write the vaHera script for you, run it, and show you the script they wrote above the result. This needs a model configured on the server (bottom edge → model).

Every step becomes a **frame**: a page that never changes afterwards. Frames line up in one continuous strip, oldest at the top, and the blank screen with the caret is always at the bottom.

## 2. Going back

Scroll up to see earlier frames. The keyboard does the same:

| Key | Goes to |
|---|---|
| PageUp / PageDown, or Alt+↑ / Alt+↓ | the previous / next frame |
| Home | the first frame |
| End | the blank screen |

A frame is a snapshot: Buhera never labels it or decides which one matters. New frames are always added at the end; nothing is overwritten. What a new step may read is the latest frame — or, once you have cut pieces onto the blank screen (§4), the frame the last piece came from.

## 3. The edges

Move the pointer to an edge of the screen and hold it there for a moment. A drawer opens; pick an item and it opens as a new frame.

| Edge | What is there |
|---|---|
| top | where you are: devices, this machine, the network, **web** (what you have read), **mail**, shared experiments, **apphub** (jobs on the university's AppHub), the runtime graph |
| right | how the screen is used: text size and spacing, screen, code visibility, printer and "save the screen as an image" |
| bottom | who the work is for: your model, projects and groups, retrieval (RAG) folders, **planning**, **spec** (understanding a specification), reports |
| left | every module, with a search at the top |

Moving the pointer to another edge switches drawers; clicking anywhere else closes it.

## 4. Many things on one screen

There are no windows to arrange. Instead you cut parts of frames and lay them on the blank screen:

- **Press the scroll wheel and drag** over any part of any frame (or hold **Alt** and drag, on a trackpad). The part you outlined lifts onto the blank screen.
- Do it again on other frames. Each piece is **live**: a plan board stays current, a mail list can still be clicked.
- Move a piece by the thin bar along its top; remove it with the × that appears when you point at it.

A morning's screen might hold your unread mail, the plan for today's experiment and the status of last night's AppHub job — three pieces from three frames, side by side.

## 5. The work verbs

These are written on the blank screen (or run here with ▶):

| You write | You get |
|---|---|
| `inbox` | your newest mail, every account |
| `mail from:mara since:2026-09-01` | a mail search ([Your Mail](./your-mail)) |
| `find plate layout blanks` | the same words looked up in what you have read, your mail, your files, the web and your plans ([Finding and Planning](./finding-and-planning)) |
| `web dcat-ap linkml` | a search engine's results |
| `read https://…` · `read site https://…` | a page, or a documentation site, read into your library |
| `diagram https://…` · `workflow https://…` · `compare https://… with https://…` | a specification's classes, its workflow, what one changes about another ([Understanding a Specification](./understanding-a-specification)) |
| `draw …` | a workflow drafted by your model from your open plan's notes |
| Mermaid (`flowchart LR …`) | drawn as written |
| `plan experiment PC 34:1 lipid series` | a new plan item |
| `plans` | the plan board |
| `apphub` | your repositories and their jobs on AppHub ([Jobs on AppHub](./jobs-on-apphub)) |
| `chart` | charts of the latest frame (`chart memory` charts what vaHera remembers) |

Try the board now — it is empty until you plan something:

```
plans
```

## 6. Preferences

Right edge → **preferences**: text size, spacing, column width, and whether pages move when they turn. Changes apply at once and are kept in this browser. Right edge → **code** decides whether the script behind each step is shown.

Next: [Your Mail](./your-mail).
