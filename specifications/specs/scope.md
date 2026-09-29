# scope — SCOPE (microscopy analysis)

| | |
|---|---|
| **Registry id** | `scope` |
| **Layer** | observation |
| **Language** | `scope` (`.scope` — a Buhera assignment; upstream examples are TypeScript strings) |
| **Upstream** | `fullscreen-triangle/helicopter` · `scope-lang/src` @ `8dd4173` |
| **Vendored at** | `long-grass/vendor/scope-lang/src` (byte-exact; `package.json`, `tsconfig.json` local) |
| **TS binding** | native — `registry-ts/src/modules/scope.ts`, bound in `long-grass/src/lib/modules/scope-module.js` |
| **Rust binding** | none — the engine is TypeScript |

## 1. Purpose

SCOPE writes a microscopy measurement as a program whose attainability is checked before any pixel is read. A program declares **channels** (synchronisation and phase cells with their actions), a **coordinate space** (field, depth, the two scale parameters), **goals** (e.g. `distance_uncertainty < 0.5 µm`, `snr > 8.0`), **rules** (conservation or symmetry invariants with a tolerance), and **morphisms** — pipelines `observe |> catalyze |> access |> measure_distance |> visualise` — plus a **dispatch** from phase cells to morphisms. The type checker refuses depth mismatches, overlapping cells, entropy budgets a rule chain exceeds, and distances measured between regions never accessed; it warns when a goal is unreachable at the declared depth (δd_min).

At run time the engine segments the image by a scale field, accesses regions, measures the distance with its uncertainty and relative uncertainty, and emits S-entropy coordinates (normalised to sum 1), goal status, SNR, the Cramér–Rao bound and channel capacity.

## 2. What is bound

`compile` (whole programs) and `createSession` (the REPL). Cells accumulate into one growing program in the module's session (R6); a cell that reaches `visualise` runs against the linked image. The session is per module instance; the host reaches it through `linkImage` and `resetSession` — long-grass's terminal decodes an image URL and links it. `load(db=…, dataset=…, image=…)` inside a program is a label: the host supplies the pixels.

`runtime/` also carries ~1.8 kLOC of legacy executors and API clients (network fetches, `NEXT_PUBLIC_HF_TOKEN`) that nothing reachable from `index.ts` imports (**U-scp-1**).

## 3. Language

```
program   := "scope" ID "{" item* "}"          -- a REPL cell may be any item on its own
item      := "channels" "{" ("sync" ID "at" NUM UNIT | "cell" ID "bounds" "(" NUM "," NUM ")" "action" ID)* "}"
           | "coordinate_space" "{" "field" NUM "x" NUM UNIT "depth" NUM "lambda_s" NUM "lambda_t" NUM "}"
           | "goal" "{" (METRIC ("<" | ">") NUM UNIT?)* "}"
           | "rule" ID "(" ID ")" "{" "invariant" ":" STRING "epsilon" ":" NUM "}"
           | ID "=" "observe" "(" (load | ID) "," "n" "=" NUM ")" ("|>" step)*
           | "dispatch" "{" ("when" ID "do" "execute" "(" ID ")")* "}"
step      := "catalyze" "(" ID "(" ID ")" ("," "confidence" "=" NUM)? ")"
           | "access" "(" ID ("," "threshold" "=" NUM)? ")"
           | "fuse" "(" ID "," NUM ")" | "measure_distance" "(" ID "," ID ")"
           | "visualise" "(" MODE ")"          -- 13 modes: scale_field, segmentation, spectral_power, …
```

`x` is a keyword (so `scope x { }` does not parse).

A goal the declared depth cannot reach is a **warning**, not an error — the program compiles and the goal chip reports ✗:

```scope
scope goal_warning_demo {
  coordinate_space {
    field 100 x 100 µm
    depth 6
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.1 µm
    distance_uncertainty < 2.0 µm
  }

  rule conservation(dna_mass) {
    invariant: "total DAPI-stained area is conserved ±5%"
    epsilon: 0.008
  }

  measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 6)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(spectral_power)
    |> visualise(uncertainty_bar)
}
```

Spindle geometry with rule confidences and access thresholds:

```scope
scope spindle_with_confidence {
  coordinate_space {
    field 64 x 64 µm
    depth 10
    lambda_s 0.08
    lambda_t 0.03
  }

  goal {
    distance_uncertainty < 0.2 µm
    relative_uncertainty < 0.02
  }

  rule symmetry(bilateral) {
    invariant: "mitotic spindle has bilateral symmetry along division axis"
    epsilon: 0.006
  }

  spindle_axis = observe(load(db="BBBC", dataset="BBBC007", image="17P1_POS0006_D_1UL.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(symmetry(bilateral), confidence = 0.8)
    |> access(nucleus_a, threshold = 0.7)
    |> access(nucleus_b, threshold = 0.7)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
}
```

## 4. Instructions

| Instruction | Effect |
|---|---|
| `string` or `{kind: "cell", source}` | evaluate one REPL cell |
| `{kind: "check", source}` | compile a whole program (no image, no run) |
| `{kind: "load", image: {data, width, height}}` | link an image |
| `"state"` / `"reset"` | what the session holds / a fresh session |

## 5. Output delta

An executing cell: `scope_run` — `{result: {structure, position, distance, uncertainty, relativeUncertainty, sEntropy{sk, st, se, sum}, goalStatus[{metric, op, threshold, actual, passed}], snr, crlbPixels, channelCapacity, chartData, visualData, log}, log}`. Other cells and controls: `text`. `visualData` holds typed arrays (images); it is large.

## 6. Residue

An executing cell: the declared goals the result did **not** meet. Defining cells, `state`, `load`, `reset`: 0. A refused cell: 1. (The former residue, `sEntropy.sum`, is 1 on every run by construction.)

## 7. Side effects and hazards

None at run time beyond CPU: a run takes on the order of a second. No network: the legacy clients are unreachable.

## 8. Conformance

- `registry-ts/test/modules.test.ts` — `scope x { }` refused at 1:7; a define–define–run REPL over a synthetic two-Gaussian image measures, meets no `snr > 1000` goal (residue 1) and shows the S-entropy sum at 1; `reset` unlinks the image.
- `long-grass/test/library-federation.test.mjs` — the canonical nuclear-separation program compiles; `scope x { }` does not.
- `long-grass/test/knowledge-packs.test.mjs` — the examples above compile.
