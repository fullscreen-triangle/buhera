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

test("a request relayed from a visitor is never local, though it arrives from loopback", () => {
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-for": "203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-for": "127.0.0.1, 203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-real-ip": "203.0.113.9" })), false);
  assert.equal(isLocalRequest(req("::1", { forwarded: "for=203.0.113.9;proto=https" })), false);
  assert.equal(isLocalRequest(req("::1", { forwarded: "for=\"[2001:db8::1]:443\"" })), false);
  assert.equal(isLocalRequest(req("::1", { forwarded: "for=\"[::1]:443\", for=203.0.113.9" })), false);
});

test("a forwarding header that cannot be read fails closed", () => {
  assert.equal(isLocalRequest(req("127.0.0.1", { forwarded: "proto=https" })), false);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-for": " , " })), false);
});

test("Next.js's own relay (forwarded for loopback) stays local", () => {
  assert.equal(isLocalRequest(req("::1", { "x-forwarded-for": "::1", "x-forwarded-host": "localhost:3000" })), true);
  assert.equal(isLocalRequest(req("127.0.0.1", { "x-forwarded-for": "127.0.0.1" })), true);
  assert.equal(isLocalRequest(req("127.0.0.1", { forwarded: "for=127.0.0.1:52100" })), true);
  assert.equal(isLocalRequest(req("::1", { forwarded: "for=\"[::1]:52100\";proto=http" })), true);
});
