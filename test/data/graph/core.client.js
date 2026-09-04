// A barrel: it re-exports its directory instead of importing it. 15% of the
// edges of the corpus are these, and a regex that only looks for `import`
// misses every one of them.
export * from './core/client/store.js'
