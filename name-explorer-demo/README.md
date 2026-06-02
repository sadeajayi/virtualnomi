# Name Explorer Demo

An interactive demo showcasing African names with their phonetic spellings, meanings, origins, and cultural stories.

## Features

- **Searchable interface**: Type any name to find matches
- **Accent-friendly search**: Works with diacritics and special characters
- **Surprise me**: Random name discovery
- **Rich name cards**: Each name shows phonetic spelling, meaning, origin, and story
- **Responsive design**: Clean, minimal interface ready for iframe embedding

## API

Name cards load from the Nomi Name Search API (configurable via `API_BASE`, default `https://nomi-name-search-api.onrender.com`):

- `GET /search?q={query}` — meaning, phonetic, language, story
- `GET /insights?name={name}&language={language}` — cultural insight paragraph (hidden if the request fails or times out)

## Running the Demo

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo (optional: point at a local API):
   ```bash
   export API_BASE="http://127.0.0.1:8000"  # optional
   python name_explorer_demo.py
   ```

3. Open your browser to `http://localhost:9000`

## Embedding in Website

This demo is designed to be embedded in your website using an iframe:

```html
<iframe 
    src="https://your-demo-url.com" 
    width="100%" 
    height="600px" 
    frameborder="0">
</iframe>
```

## Customization

- Set `API_BASE` to your deployed API hostname
- Modify the CSS in the `css` parameter to match your brand
- Update the CTA link to point to your submission form
