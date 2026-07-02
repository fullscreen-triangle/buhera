/* ============================================================================
 * Purpose Module Adapter
 *
 * Wraps the JS federation in src/lib/purpose/ (federation + llm + knowledge
 * packs + storage) as a Buhera module. The module is a "synthesis" primitive:
 * given a description and optional followups, it runs a knapsack-allocated
 * cascade of federated models and returns a synthesis document.
 *
 * v1 instruction shape (accepts either):
 *   • { kind: "synthesise", description, followups?, field? }
 *   • a plain string — treated as the description with no followups
 *
 * v1 also carries a graceful degradation path: if no LLM provider is
 * configured (no HUGGINGFACE_API_KEY / ANTHROPIC_API_KEY / LLM_PROVIDER),
 * the module returns ok:false with a clear message pointing to .env.local.
 * ========================================================================== */

// The purpose lib exposes the primitives; we compose one straight-shot
// synthesis call here. Real multi-turn / storage flows can be added later
// as additional instruction kinds.
import { getProvider, synthesisModel } from "../purpose/llm.js";
import { selectPacks, buildPackContext } from "../purpose/knowledge-packs.js";
import {
  getFederationModels,
  aggregateFloor,
  federationMetadata,
} from "../purpose/federation.js";

function providerAvailable() {
  try {
    getProvider();
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

function normaliseInstruction(instruction) {
  if (instruction == null || instruction === "") {
    return { description: "", followups: [], field: null };
  }
  if (typeof instruction === "string") {
    return { description: instruction, followups: [], field: null };
  }
  if (typeof instruction === "object" && instruction.kind === "synthesise") {
    return {
      description: String(instruction.description || ""),
      followups: Array.isArray(instruction.followups)
        ? instruction.followups.map(String)
        : [],
      field: instruction.field || null,
    };
  }
  if (typeof instruction === "object") {
    return {
      description: String(instruction.description || ""),
      followups: Array.isArray(instruction.followups)
        ? instruction.followups.map(String)
        : [],
      field: instruction.field || null,
    };
  }
  return { description: String(instruction), followups: [], field: null };
}

export const purposeModule = {
  id: "purpose",

  describe() {
    return {
      id: "purpose",
      description:
        "Purpose: federated knapsack-allocated cascade over LLMs. " +
        "Takes a description, returns a synthesis document.",
      instructions: [
        'dispatch("purpose", "your research description")',
        'dispatch("purpose", { kind: "synthesise", description: "...", followups: ["..."] })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const provider = providerAvailable();
    if (!provider.ok) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            "purpose: no LLM provider configured.",
            "set HUGGINGFACE_API_KEY or ANTHROPIC_API_KEY in .env.local",
            `(details: ${provider.error})`,
          ],
        },
        residue: 0,
        completed: true,
        error: provider.error,
      };
    }

    const { description, followups } = normaliseInstruction(instruction);
    if (!description.trim()) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: ["purpose: instruction has no description text."],
        },
        residue: 0,
        completed: true,
      };
    }

    try {
      // Compose the prompt: description + any followups, plus the
      // knowledge-pack context most relevant to the query.
      const haystack = [description, ...followups].join("\n");
      const packs = selectPacks(haystack, { budget: 60_000 });
      const packContext = buildPackContext(packs.map((p) => p.id));

      const prov = getProvider();
      const model = synthesisModel();

      const userText = [
        packContext ? `Context:\n${packContext}\n` : "",
        `Description:\n${description}`,
        followups.length
          ? `\nFollowups:\n${followups.map((f) => `- ${f}`).join("\n")}`
          : "",
      ]
        .join("")
        .trim();

      const synthesis = await prov.chat({
        system:
          "You are Purpose, a federated research synthesis engine. " +
          "Produce a concise, well-cited synthesis document.",
        messages: [{ role: "user", content: userText }],
        model,
        maxTokens: 2048,
      });

      const federationIds = getFederationModels();
      const floor = aggregateFloor([...federationIds, model]);
      const meta = federationMetadata(federationIds);

      return {
        ok: true,
        output_delta: {
          kind: "purpose_synthesis",
          synthesis,
          model,
          federation: meta,
          floor,
          packs_used: packs.map((p) => ({ id: p.id })),
        },
        residue: floor,
        completed: true,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`purpose error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  outputCell(_instruction) {
    return { kind: "synthesis_cell" };
  },
};
