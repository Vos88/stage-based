# Project info

Link to GitHub pages [AI-flow-guide](https://vos88.github.io/ai-flow-guide/).

## Project setup

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS
- KaTeX (for LaTeX math rendering)

## LaTeX Math Rendering

This project includes LaTeX math rendering capabilities using KaTeX. Mathematical expressions can be written in the content using:

- **Inline math**: Use single dollar signs `$...$` for inline mathematical expressions
- **Block math**: Use double dollar signs `$$...$$` for displayed mathematical equations

The LaTeX rendering is automatically applied to all content sections including overview, how it works, applications, advantages, and limitations.

## Project deployment

This repo is already configured to deploy to GitHub pages.  
To setup your forked repo, prerequisites are required.

### Prerequisits

- Installed `gh-pages` package.
- Pages enabled in GitHub repo: Repository -> Settings -> Pages -> Deploy from: gh-pages branch (create this branch).

The `vite.config.ts` needs to point towards the `dist` folder, a folder that gh-pages creates to succesfully deploy the project. In `vite.config.ts` ensure that base is set with the project name:

- `base: mode == 'pages' ? '/{project_name}/'`.

Additionally, in `index.html` the `<BrowserRouter basename>` needs to be set to:

- `<{window.location.hostname.includes('github.io') ? "project_name" : "/"}>`

To deploy the project, run the following commands in any terminal

```Powershell
# Step1: build the pages
npm build:pages

# Step 2: deploy to pages
npm run deploy
```

If npm does not recogise commands, add to `package.json`:

```python
  "scripts": {
    "build:pages": "vite build --mode pages",
    "deploy": "gh-pages -d dist"
  },
```
