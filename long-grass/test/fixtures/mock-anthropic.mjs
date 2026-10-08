// A stand-in for the Messages API: the first request is answered with a
// web_search tool call, the second with text. Records what it was sent.
import http from "http";

const sse = (events) => events.map(([type, data]) => `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`).join("");
const start = { message: { id: "m", type: "message", role: "assistant", model: "claude-opus-5-5", content: [], stop_reason: null, usage: { input_tokens: 10, output_tokens: 1 } } };

export function mockAnthropic() {
  const requests = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      requests.push({ url: req.url, headers: req.headers, body: JSON.parse(body) });
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      const events = requests.length === 1
        ? [["message_start", start],
           ["content_block_start", { index: 0, content_block: { type: "tool_use", id: "tu1", name: "web_search", input: {} } }],
           ["content_block_delta", { index: 0, delta: { type: "input_json_delta", partial_json: "{\"query\": \"DCAT-AP-PLUS\"}" } }],
           ["content_block_stop", { index: 0 }],
           ["message_delta", { delta: { stop_reason: "tool_use" }, usage: { output_tokens: 5 } }],
           ["message_stop", {}]]
        : [["message_start", start],
           ["content_block_start", { index: 0, content_block: { type: "text", text: "" } }],
           ["content_block_delta", { index: 0, delta: { type: "text_delta", text: "DCAT-AP+ adds provenance " } }],
           ["content_block_delta", { index: 0, delta: { type: "text_delta", text: "([source](https://nfdi-de.github.io/dcat-ap-plus/latest/))." } }],
           ["content_block_stop", { index: 0 }],
           ["message_delta", { delta: { stop_reason: "end_turn" }, usage: { output_tokens: 9 } }],
           ["message_stop", {}]];
      res.end(sse(events));
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, requests, url: `http://127.0.0.1:${server.address().port}` })));
}
