# NIDS frontend (React + Vite)

Dashboard for the NIDS backend: upload a capture, watch progress, review flagged flows, download predictions.

## Setup

1. Start the backend first (see `../backend/README.md`), default `http://localhost:5000`.
2. `npm install`
3. `npm run dev` and open `http://localhost:5173`.

The dev server proxies `/api` to the backend. If the backend runs elsewhere, copy `.env.example` to `.env`
and set `BACKEND_URL`. For production, `npm run build` and serve `dist/` behind the same host as the API
(or add a reverse proxy for `/api`).

## Files

- `src/App.jsx` - upload, progress steps, class breakdown, flow table
- `src/styles.css` - styling
