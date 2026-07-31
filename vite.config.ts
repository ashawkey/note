import { defineConfig } from 'vite'

/*
 * Vite's only job here is the shell bundle — the small amount of client code (theme
 * toggle, search modal, sidebar drawer) plus our own CSS layers.
 *
 * The pages themselves are written by scripts/build.ts, which reads the manifest below to
 * learn the hashed filenames. Nothing about the document rendering runs through Vite.
 *
 * The document stylesheets in src/styles/ are deliberately *not* bundled. They are
 * concatenated into dist/theme.css by src/site/theme.ts, and the webfonts stay in public/ so
 * their relative url() references keep resolving against their own stylesheet.
 */
export default defineConfig({
  // Assets are referenced with paths computed per page depth, so the built site works at a
  // domain root, a project-page subpath, or straight off disk with no rebuild.
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    manifest: true,
    cssMinify: true,
    rollupOptions: {
      input: 'src/shell/shell.ts',
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
