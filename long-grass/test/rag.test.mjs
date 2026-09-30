// Tests for the RAG route's locality guard: only a direct loopback request
// may name folders for the server to read; a proxied one never may.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isLocalRequest } from "../src/lib/server/rag.js";

const req = (remoteAddress, headers = {}) => ({ socket: { remoteAddress }, headers });

test("a loopback request with no proxy headers is local", () => {
  assert.equal(isLocalRequest(req("127.0.0.1")), true);
  assert.equal(isLocalRequest(req("::1")), true);
  assert.equal(isLocalRequest(req("::ffff:127.0.0.1")), true);
});

test("a remote request is not local", () => {
  assert.equal(isLocalRequest(req("203.0.113.9")), false);
});

test("a request relayed by a reverse proxy is never local, though it arrives from loopback", () => {
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-for": "203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-real-ip": "203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("::1", { forwarded: "for=203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-host": "example.org" })), false);
});
