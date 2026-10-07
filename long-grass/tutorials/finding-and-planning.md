# Finding and Planning

`find` looks for something everywhere at once — what you have read, your mail, your files, the web, your own plans — and shows each answer under the place it came from, with that place's own honesty about how good the match is. `plan` turns what you found into an experiment or a task: steps, the evidence you kept and how sure each search was, and the AppHub jobs that run it.

**Time:** 20 minutes.

**Before you start:** [Your Mail](./your-mail), with at least one account, ideally with mail kept (§4 there). "Your files" is the folder the server searches: on your own computer, the repository long-grass sits in, or the folder named by `SPRAYPAINT_ROOT` — a folder of lab notes is a good choice.

---

## 1. Find

Write what the answer would contain — words, not a question:

```
find internal standard blanks
```

Run against a test mailbox and a small folder of lab notes (an extraction protocol, a meeting note), before anything had been read from the web, it gave:

```text
found for "internal standard blanks" — previews: nothing is committed until you keep it.

YOUR MAIL
2026-10-02  Re: LARA run   Mara Lind   uni   + keep on a plan
in the kept mail, with a verdict
  covered — uni/INBOX/2026-10/…-5.md:1-8 contains every query term
  8 │ Thursday works. Bring the blanks & the internal standard.
  …-2.md:11-14   matched internal, standard
  12 │ two replicates per plate. Let me know about the internal standard.

YOUR FILES
  covered — protocols/lipid-extraction.md:1-8 contains every query term
  3 │ 1. Add 300 µL methanol with the internal standard (PC 17:0/17:0, 10 µM) to each well.

YOUR PLANS
  no plan of yours mentions it.

THE WEB
  duckduckgo · the pages themselves are not read until you read them
  (ten results, each with read and + keep on a plan)
```

- **Your mail** is searched live in every account, and — once mail is kept — once more with a verdict.
- **Your files** are searched by spraypaint: passages ranked by the words in them, each shown as the few lines where your words are densest, numbered as in the file.
- **What you have read** — every page kept with `read` ([Understanding a Specification](./understanding-a-specification)) — is searched with a verdict, like your files. Before you have read anything, it says so.
- **The web** is a search engine: DuckDuckGo, with no key needed (a server can be pointed at its own SearXNG with `SEARXNG_URL`, or Brave with `BRAVE_SEARCH_KEY`). A result is the engine's title and snippet — a pointer, not a reading. **read** keeps the page; then it is searched with a verdict too.
- **Your plans** are the plan items whose title, notes, steps or kept evidence contain every word.

Each source is asked on its own, so one failing does not hide the others.

## 2. What a verdict lets you say

A search over words always returns something that shares a word with your query. The verdict says whether that something is about what you asked:

| Verdict | Means | You may say |
|---|---|---|
| **covered** | one passage holds every word | "this passage uses all these words" — then read it |
| **partial** | the words occur, never all in one passage | "the pieces are here, scattered"; split the question |
| **declined** | the words you asked about are not there | "my mail / my notes do not mention this, in these words" |

```
find suitable dentist
```

```text
YOUR MAIL
  no message matches, in any account.
  declined — no returned passage contains any query term; not in the corpus: suitable, dentist
YOUR FILES
  declined — no returned passage contains any query term; not in the corpus: suitable, dentist
```

Declined is not "it does not exist" — it is "not in what was searched, in these words". Try another word for it (Zahnarzt), or look on the web. And covered is not "answered": it means the words are together, so read the passage before you rely on it.

## 3. Plan

```
plan experiment PC 34:1 lipid series on LARA
```

A plan item opens, in the active project (bottom edge → projects). An **experiment** starts with the steps experiments usually have; a **task** (`plan task …`, or just `plan …`) starts empty:

```text
PC 34:1 lipid series on LARA
experiment   idea · planned · running · done · dropped   due __.__.____

STEPS
  ☐ state the question
  ☐ find what is already known — mail, notes, papers
  ☐ materials and method
  ☐ book time: instrument, AppHub session
  ☐ run
  ☐ analyse
  ☐ write the report

WHAT WE FOUND        find for this →
JOBS ON APPHUB
```

Everything on it is edited in place: the title, the status, the due date, notes, steps (tick, add, remove). **find for this →** runs `find` on the item's title.

## 4. Keep what you found

Back on a `find` frame, every message and passage has **+ keep on a plan**: choose the item, or type a name to start a new one. The item now lists it:

```text
WHAT WE FOUND
mail  Re: LARA run                                   open  remove
      mail:uni/INBOX/5
      from Mara Lind, 2026-10-02
      found by "internal standard blanks"
mail  attached is the sample prep protocol for the…  covered  committed act #1  open  remove
      kept mail: uni/INBOX/2026-09/…-2.md:11-14
      found by "internal standard blanks"
```

Each kept thing records where it is (`cite`), the words that found it, and — for a search that gives one — its verdict. Searching is free: every `find` is a preview. Keeping a passage is relying on that search, so it is committed then, once, and the item shows the act number; spraypaint's count of committed acts only ever goes up, so it records the answers you used, not the searches you tried.

## 5. The board

```
plans
```

```text
project default · 1 item
IDEA
  PC 34:1 lipid series on LARA   experiment   0/7 steps   2 found
```

Items are grouped by status, open ones first. Click one to open it. Each item has **save as markdown** — the plan, its steps, what was found with its citations and verdicts, and its jobs, as a file you can send or print.

Plans are kept in this browser, per project. Cut the board onto your blank screen (wheel-press and drag) and it stays current there.

Next: [Jobs on AppHub](./jobs-on-apphub) — running the scoring step of this experiment on the university's machines.
