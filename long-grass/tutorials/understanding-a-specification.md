# Understanding a Specification: DCAT-AP and DCAT-AP+

Two specifications to understand, on the blank screen: DCAT-AP 3.0.1, the European profile for describing datasets in catalogues, and DCAT-AP+, which extends it with how a dataset came to be — the experiment, the instrument, the plan. You will find and read both, take notes as you go, ask what you have read, and draw them: a class diagram, the workflow each one describes, and what the second changes about the first. Everything ends up in one task you can save as Markdown.

**Time:** 45 minutes.

**What you end with:** a task holding your notes (each with the passage and its address), the diagrams, a comparison, and a Markdown file of all of it.

**Before you start:** nothing to install. On the hosted site, sign in first; reading the web there is for members only. Every example output below is from a real run on 7 October 2026, against DCAT-AP 3.0.1 (published 27 October 2025) and DCAT-AP+ 0.1.0rc4.

---

## 0. What Buhera does here, and what it does not

| It does | It does not |
|---|---|
| read the pages you name, and keep them in your library | read anything you did not ask for |
| draw class diagrams and workflows from what a specification **states** — its property tables, its schema | guess a diagram from the prose |
| compare two specifications class by class, property by property | decide which one is right |
| let your model draft a diagram from **your notes**, and label it as the model's | present a drafted diagram as fact |

Diagrams drawn from a specification are only as complete as what it states in tables or a schema. A sentence in a usage note that the tables do not repeat will not appear in them; read the page for those.

## 1. A task to hold what you learn

```
plan task understand DCAT-AP 3.0.1 and DCAT-AP+
```

A plan item opens. Notes you take while reading land here, with the passage they are about.

## 2. Find the pages

You may already have the addresses. If not, search — with a search engine, no account or key needed:

```
web DCAT-AP-PLUS LinkML
```

```text
web "DCAT-AP-PLUS LinkML" · 10 results · duckduckgo
GitHub - nfdi-de/dcat-ap-plus: A domain-agnostic extension of the DCAT ...   read  + keep on a plan
https://github.com/nfdi-de/dcat-ap-plus
DCAT-AP Plus Links to Use-case Specific Context (DCAT-AP+)                    read  + keep on a plan
https://nfdi-de.github.io/dcat-ap-plus/latest/
```

A result is a pointer, not a reading: the snippet is the search engine's, and nothing is kept until you **read** the page.

## 3. Read DCAT-AP

```
read https://semiceu.github.io/DCAT-AP/releases/3.0.1/
```

```text
DCAT-AP 3.0.1
https://semiceu.github.io/DCAT-AP/releases/3.0.1/   read 2026-10-07   17,822 words   82 sections
[class diagram] [workflow] [what it defines]   + keep on a plan
▾ OUTLINE
  Abstract · Introduction · Conformance Statement · Terminology · Overview
  Main Entities: Agent, Catalogue, Catalogue Record, …, Dataset, Distribution, …
  Supportive Entities: Activity, Attribution, …   Datatypes · Controlled Vocabularies · …
```

The page is kept in your library as text — navigation, scripts and decoration dropped, the property tables kept as tables. The outline jumps to a section. The specification itself is long; read the **Conformance Statement** and **Dataset** first.

## 4. Take notes

Beside every section heading there is **+ note**. Write what the passage says in your own words, choose the task, **keep the note** (Ctrl+Enter). The note is kept with the passage and its address, so it can always be checked:

```text
noted on "understand DCAT-AP 3.0.1 and DCAT-AP+"

Dataset — your note:
  A Dataset needs only a title and a description (both 1..n). Everything else —
  distributions, publisher, and how it was generated (prov:wasGeneratedBy → Activity,
  0..*) — is optional.
  — https://semiceu.github.io/DCAT-AP/releases/3.0.1/#Dataset
```

Several notes may cite the same section. A note is yours: Buhera does not check it against the page — the address is there so you can.

## 5. Read DCAT-AP+ — a whole documentation site

DCAT-AP+ is documented across many pages, generated from its LinkML schema. Read the site, not just its front page:

```
read site https://nfdi-de.github.io/dcat-ap-plus/latest/
```

```text
read 30 pages under https://nfdi-de.github.io/dcat-ap-plus/latest/ · 613 more linked pages left unread (limit 30)
DCAT-AP+ Documentation                                   429 words
Design Patterns - DCAT-AP+ Documentation               4,170 words
Extending Rules · Automatic Generation · Schema · Versioning · Users · About
dcat_ap_plus.yaml                                      8,255 words
dcat_ap_linkml.yaml                                    6,129 words
Class: Activity · DataGeneratingActivity · DataAnalysis · EvaluatedActivity · Device · Dataset · …
```

It starts at the address and follows links under the same path, breadth first: the narrative pages and the two schemas come first, then the generated class pages. `read site <address> 80` reads up to 80. Two pages matter most: **Design Patterns** (why the extension is shaped as it is) and **dcat_ap_plus.yaml** (the schema itself — the documentation is generated from it).

## 6. Ask what you have read

Everything read is searchable together, with a verdict on whether it covers your words:

```
find dataset generating activity plan instrument
```

```text
WHAT YOU HAVE READ
covered — …/elements/overview/index.md:1-40 contains every query term
  …/elements/overview   matched dataset, generating, activity, plan, instrument   open Schema - DCAT-AP+ Documentation
    8 │ This metadata schema is an Extension of the DCAT Application Profile for Providing Links to
      │ Use-case Specific Context. It allows to provide additional metadata regarding: which kind(s) …
  …/schema/dcat_ap_plus.yaml   matched dataset, generating, activity, plan   open dcat_ap_plus.yaml
   16 │ kind of instruments were used in the dataset generating activity, in which surrounding
   17 │ (e.g. a laboratory) and according to which plan the dataset generating activity
```

`find` also looks in your mail, your files, the web and your plans — see [Finding and Planning](./finding-and-planning). **Covered** means one passage holds all your words; it does not mean the passage answers your question. Read it.

## 7. What DCAT-AP defines

On the DCAT-AP page, press **what it defines**:

```text
DCAT-AP 3.0.1 · 33 classes, 131 properties · read from its property tables
class          properties   mandatory
Agent                   2   name
Catalogue              19   description, publisher, title
Catalogue Record        9   modification date, primary topic
Data Service           16   endpoint URL, title
Dataset                36   description, title
Distribution           24   access URL
…
```

These are the specification's own tables, read row by row: a property is mandatory when its minimum cardinality is 1. 33 classes is its 13 main and 21 supportive entities, less Literal, which is a datatype.

## 8. A class diagram

```
diagram https://semiceu.github.io/DCAT-AP/releases/3.0.1/
```

Dataset in the middle, every class it points to, each arrow labelled with the property and its cardinality, mandatory data properties inside the boxes. Above the diagram: **around** another class, depth **1** or **2**, properties **mandatory / all / none**. Each change is a new frame, so you can scroll back to compare. Under it: **show the Mermaid** (the diagram as text), **copy**, **save as SVG**, **+ keep on a plan**.

```
diagram https://semiceu.github.io/DCAT-AP/releases/3.0.1/ around Distribution
```

## 9. The workflow each specification describes

DCAT-AP+ is about how a dataset came to be — a workflow — and it states that workflow in PROV-O terms: what an activity used, who or what it was carried out by, where, and what it generated. Buhera draws exactly those statements:

```
workflow https://nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml against https://semiceu.github.io/DCAT-AP/releases/3.0.1/
```

```text
the workflow the specification's PROV-O terms state · teal: added by this specification (19 classes)

  Plan ──realized plan──▶ (DataGeneratingActivity) ══was generated by══▶ Dataset
  EvaluatedEntity ──evaluated entity──▶ (DataGeneratingActivity)
  {AgenticEntity} ┄carried out by┄▶ (Activity)      [Surrounding] ┄occurred in┄ (DataGeneratingActivity)
  (DataGeneratingActivity) ══was generated by══▶ AnalysisSourceData ──evaluated entity──▶ (DataAnalysis)
  (DataAnalysis) ══was generated by══▶ AnalysisDataset
```

Activities are rounded, agents and instruments are hexagons, plans parallelograms, places cylinders. `against` the base colours what DCAT-AP+ adds. Now the base on its own:

```
workflow https://semiceu.github.io/DCAT-AP/releases/3.0.1/
```

```text
(Activity) ══was generated by══▶ Dataset
```

One edge. In DCAT-AP a Dataset may say an Activity generated it, and the Activity has no properties of its own — the DCAT-AP+ documentation calls it "a dead end". The two diagrams side by side are the whole motivation for DCAT-AP+. Keep both on your task (**+ keep on a plan**).

## 10. What DCAT-AP+ changes about DCAT-AP

```
compare https://semiceu.github.io/DCAT-AP/releases/3.0.1/ with https://nfdi-de.github.io/dcat-ap-plus/latest/schema/dcat_ap_plus.yaml
```

```text
what DCAT-AP-PLUS 0.1.0rc4 changes about DCAT-AP 3.0.1
  19 classes added   58 properties added   1 made required   2 ranges narrowed

WHAT IT CHANGES IN PROPERTIES THE BASE ALREADY HAD
Dataset
  was_generated_by: 0..* → 1..* — now required
  was_generated_by: range Activity → DataGeneratingActivity (a subclass)
CatalogueRecord
  primary_topic: range CataloguedResource → Any

CLASSES IT ADDS
  AgenticEntity · AnalysisDataset (is a Dataset) · AnalysisSourceData · DataAnalysis
  DataGeneratingActivity (is a Activity) · Device (is a AgenticEntity) · EvaluatedEntity
  EvaluatedActivity · Plan · QualitativeAttribute · QuantitativeAttribute · Software · Surrounding · …
```

The centre of the extension, in one line: **every Dataset must now say which data-generating activity produced it**, and that activity has a plan, an evaluated entity, an instrument and a place. The 58 added properties are mostly title and description on supportive classes, and the twelve properties that give Activity its content (`had_input_entity`, `carried_out_by`, `has_quantitative_attribute`, …) — open them under **properties it adds, by class**.

Classes are matched by name and properties by URI. A match the two specifications would not agree on is possible, so each change links back to the base's own row (**base**). Check a surprising one.

## 11. Your own workflow diagram

The diagrams so far are what the specifications state. For your own experiment, write the workflow yourself — Mermaid, pasted on the blank screen, is drawn as written. Here, the PC 34:1 lipid series from the [running example](./your-mail) in DCAT-AP+'s terms:

```
flowchart LR
  plan[/"Plan: PC 34:1 series protocol"/] -->|realized plan| run(["DataGeneratingActivity: LARA run"])
  sample["EvaluatedEntity: lipid sample"] -->|evaluated entity| run
  lara{{"Device: LARA robot"}} -.->|carried out by| run
  lab[("Surrounding: the lab")] -.-|occurred in| run
  run ==>|was generated by| raw["AnalysisSourceData: raw spectra"]
  run ==>|was generated by| record["Dataset: the catalogue record"]
  raw -->|evaluated entity| scoring(["DataAnalysis: chain-length scoring on AppHub"])
  scoring ==>|was generated by| scores["AnalysisDataset: scores"]
```

Or ask your model to draft one from your task's notes:

```
draw the workflow by which a lab experiment becomes a DCAT-AP+ dataset
```

The draft is checked to parse as Mermaid (and sent back to the model to correct, up to twice), then drawn under “drafted by your model … — check it against your sources”. A small local model draws plausible boxes that may not match the specification; treat a draft as a starting point to edit (**show the Mermaid**, copy, change, paste back). Drafting needs a model on the server (bottom edge → model).

## 12. Keep it, and take it with you

```
plans
```

Open **understand DCAT-AP 3.0.1 and DCAT-AP+**. Under “what we found”: your notes, each with its passage and address; the diagrams, drawn; the comparison with its summary. **save as markdown** writes it all to one file — notes under “Notes”, diagrams as Mermaid code blocks under “Diagrams”, which GitHub, GitLab and most editors draw — ready to send or to keep beside your work.

Next: [The Blank Screen](./the-blank-screen) for everything else the surface does, then [Your Mail](./your-mail).
