# @four-sided-triangle/individuate — vendored

Source: fullscreen-triangle/four-sided-triangle, `sdk-ts/` at commit 6090e51
(local clone C:\Users\kunda\Documents\semantics\four-sided-triangle).

`dist/` was compiled from that commit's `src/` with the package's own
`tsc -p tsconfig.json` (the clone's checked-out `dist/` predated
`src/receiver.ts`). `src/` is copied alongside for reading. Zero runtime
dependencies; Node-only (`LocalFileSource` and `JsonFilePersistAdapter` use
`fs`), so long-grass runs it server-side, in /api/player.

Refresh: rebuild from a newer commit the same way and update the hash above.
