// Catalogue conformance for the TypeScript federation (specification 05),
// with every real engine injected.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { conformance, type Catalogue } from "../src/index.ts";
import { createFederation } from "../src/modules/index.ts";
import { realEngines } from "./engines.ts";

test("the TS federation conforms to the catalogue (C1–C5)", async () => {
  const cat = JSON.parse(
    await readFile(new URL("../../../specifications/registry/catalogue.json", import.meta.url), "utf8"),
  ) as Catalogue;
  const { registry, dsls } = createFederation(await realEngines());
  assert.deepEqual(conformance(cat, "ts", registry, dsls), []);
});
