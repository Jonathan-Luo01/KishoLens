# KishoLens Frontend

Astro + React frontend for the KishoLens prose analysis dashboard.

## Setup

```bash
npm install
npm run dev      # dev server on http://localhost:4321
npm run build    # production build
npm run preview  # preview production build
```

## Stack

- **Astro** — static site framework and page routing
- **React** — interactive chart islands (`client:visible`)
- **D3.js / Chart.js** — data visualisation (added per feature)
- **Vanilla CSS** — scoped styles with CSS custom properties

## Folder Structure

```
src/
  pages/       # Astro pages and routes
  components/  # Astro components and React islands
    react/     # Interactive chart widgets
  styles/      # Global CSS tokens and utilities
```
