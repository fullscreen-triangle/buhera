// Ambient shim so the package typechecks standalone.
//
// `api-clients.ts` reads `process.env.NEXT_PUBLIC_HF_TOKEN`, a Next.js
// build-time-inlined variable available in the host app. It is off the
// compile()/runScope() critical path; this declaration only satisfies the
// standalone typecheck without depending on @types/node.
declare const process: {
  env: Record<string, string | undefined>;
};
