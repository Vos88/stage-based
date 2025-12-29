# Project info

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

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

### Examples: # noqa:MD026

```latex
Inline math: The slope is $\\beta_1 = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum(x_i - \\bar{x})^2}$

Block math: 
$$y = \\beta_0 + \\beta_1 x + \\varepsilon$$
```

The LaTeX rendering is automatically applied to all content sections including overview, how it works, applications, advantages, and limitations.
