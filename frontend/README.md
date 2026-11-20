# Frontend

Next.js chat UI for the `conversational_toolkit` API. It talks to the backend at `SERVER_URL`, set in `.env` to `http://localhost:8080` (the default port of `ki_kompass_rag.main`).

```bash
npm install
npm run dev        # http://localhost:3000
npm run dev:mock   # mock services, no backend needed
npm run lint
npm run build      # static export to dist/
```

`npm run dev` first runs `open http://localhost:3000`, which exists only on macOS; elsewhere run `npx next dev` and open the URL yourself.

`npm run build` sets `BUILD_MODE=export` and writes a static site to `dist/`. The toolkit's `create_app(controller, dist_path=...)` can serve that folder.

To adapt the UI:

- `public/`: logo and favicon
- `src/styles/globals.css`: theme colours and styles
- `src/config.ts`: app name, description, logo path and agent name
- `src/lib/lang/`: UI text and prompt suggestions (en, de, fr, it)
