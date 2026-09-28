/* ============================================================================
 * The DSL registry — specification 04-dsl-registry.md.
 *
 * Language id → { real validator, executing module, grounding pack }. A
 * validator must call the language's OWN front end and normalise its verdict
 * to { ok, errors:[{message, line?, column?}] }. It must be pure and must not
 * execute the program.
 * ========================================================================== */

import { errorText } from "./contract.ts";

export interface DslError {
  message: string;
  line?: number;
  column?: number;
}

export interface Validation {
  ok: boolean;
  errors: DslError[];
}

export type Validator = (source: string) => Validation;

export interface DslEntry {
  id: string;
  label: string;
  extension: string;
  moduleId: string;
  packId: string;
  validate: Validator;
}

export type DslSummary = Omit<DslEntry, "validate">;

export class UnknownDslError extends Error {
  constructor(id: string) {
    super(`unknown DSL: "${id}"`);
    this.name = "UnknownDslError";
  }
}

export class DslRegistry {
  #entries = new Map<string, DslEntry>();

  register(entry: DslEntry): DslEntry | null {
    const prev = this.#entries.get(entry.id) ?? null;
    this.#entries.set(entry.id, entry);
    return prev;
  }

  get(id: string): DslEntry | null {
    return this.#entries.get(id) ?? null;
  }

  ids(): string[] {
    return [...this.#entries.keys()].sort();
  }

  list(): DslSummary[] {
    return this.ids().map((id) => {
      const { validate: _v, ...rest } = this.#entries.get(id) as DslEntry;
      return rest;
    });
  }

  /** Throws UnknownDslError for an unregistered id (a programming error). */
  validate(id: string, source: string): Validation {
    const e = this.#entries.get(id);
    if (!e) throw new UnknownDslError(id);
    return e.validate(source);
  }

  byExtension(ext: string): DslEntry | null {
    const want = ext.toLowerCase();
    for (const e of this.#entries.values()) if (e.extension.toLowerCase() === want) return e;
    return null;
  }
}

export const valid = (): Validation => ({ ok: true, errors: [] });

export function invalidSource(errors: DslError[]): Validation {
  return { ok: false, errors: errors.length ? errors : [{ message: "rejected" }] };
}

/** 1-based line from a message of the form "line N: …", or undefined. */
export function lineFromMessage(message: string): number | undefined {
  const m = /(?:^|\b)line\s+(\d+)\b/i.exec(message || "");
  return m ? Number.parseInt(m[1] as string, 10) : undefined;
}

/** Wrap a front end that throws on the first error. */
export function fromThrowing(parse: (src: string) => unknown): Validator {
  return (src) => {
    try {
      parse(src);
      return valid();
    } catch (err) {
      const message = errorText(err);
      const line = lineFromMessage(message);
      return invalidSource([line === undefined ? { message } : { message, line }]);
    }
  };
}
