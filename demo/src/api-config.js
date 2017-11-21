/**
 * The backend always runs on port 8000. In production we also
 * serve the frontend from there. However, for development
 * we want to `npm run serve` the unminified js on port 3000.
 * This allows us to get the correct API root either way.
 */

let apiRoot;

const origin = window && window.location && window.location.origin;

if (origin === 'http://localhost:3000') {
    apiRoot = 'http://localhost:8000';
} else {
    apiRoot = origin;
}

export const API_ROOT = apiRoot;
