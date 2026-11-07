# OlfactAI Frontend

## Overview
This directory is reserved for the future OlfactAI web frontend application.

## Planned Technology Stack
- **Framework:** Next.js (App Router)
- **Library:** React 19 / TypeScript
- **Styling:** Tailwind CSS & shadcn/ui
- **Visualizations:** Recharts, Plotly.js, 3D molecular canvas (e.g. 3Dmol.js / SmilesDrawer)
- **State Management:** TanStack Query & Zustand

## Integration Architecture
The frontend will communicate with the backend via the REST API at `/api/v1`:
- Molecular structure input & 2D/3D visualization
- Smell profile prediction cards & radar charts
- Molecular similarity search & chemical space exploration
- Virtual electronic nose sensor array simulation
- Odor mixture prediction & ratio blending

Implementation will commence in a future project phase following core ML and chemistry engine milestones.

