# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/b387b741-e4c4-4874-be5c-6c3f44deaafa

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/b387b741-e4c4-4874-be5c-6c3f44deaafa) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

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

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

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

### Examples:

```latex
Inline math: The slope is $\\beta_1 = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum(x_i - \\bar{x})^2}$

Block math: 
$$y = \\beta_0 + \\beta_1 x + \\varepsilon$$
```

The LaTeX rendering is automatically applied to all content sections including overview, how it works, applications, advantages, and limitations.

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/b387b741-e4c4-4874-be5c-6c3f44deaafa) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/tips-tricks/custom-domain#step-by-step-guide)
