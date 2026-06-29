import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    base: '/knowledge/',
    title: 'knowledge',
    description: 'AI developer assistant for the Kalisio ecosystem',
    ignoreDeadLinks: true,
    // docs/experiments/ holds the Python experiment code, notebooks (.ipynb) and the old
    // project README — none of it is rendered. The write-ups now live under research/.
    srcExclude: ['**/experiments/**'],
    head: [
      ['link', { href: 'https://cdnjs.cloudflare.com/ajax/libs/line-awesome/1.3.0/line-awesome/css/line-awesome.min.css', rel: 'stylesheet' }],
      ['link', { rel: 'icon', href: `/images/favicon.ico` }]
    ],
    themeConfig: {
      logo: '/images/knowledge-logo.png',
      socialLinks: [{ icon: 'github', link: 'https://github.com/kalisio/knowledge' }],
      nav: [
        { text: 'About', link: '/about/introduction' },
        { text: 'Guides', link: '/guides/understanding' },
        { text: 'Architecture', link: '/architecture/introduction' },
        { text: 'Research', link: '/research/introduction' }
      ],
      sidebar: {
        '/about/': getAboutSidebar(),
        '/guides/': getGuidesSidebar(),
        '/architecture/': getArchitectureSidebar(),
        '/research/': getResearchSidebar()
      },
      footer: {
        copyright: 'MIT Licensed | Copyright © 2017-20xx Kalisio'
      }
    },
    vite: {
      optimizeDeps: {
        include: ['keycloak-js', 'lodash']
      },
      build: {
        commonjsOptions: {
          include: [/node_modules/]
        }
      },
      ssr: {
        noExternal: ['vitepress-theme-kalisio']
      }
    }
  })
)

function getAboutSidebar () {
  return [
    { text: 'Introduction', link: '/about/introduction' },
    { text: 'Contributing', link: '/about/contributing' },
    { text: 'License', link: '/about/license' },
    { text: 'Contact', link: '/about/contact' }
  ]
}

function getGuidesSidebar () {
  return [
    { text: 'Understanding knowledge', link: '/guides/understanding' },
    { text: 'Getting started', link: '/guides/getting-started' },
    { text: 'Semantic code search', link: '/guides/semantic-search' },
    { text: 'Git intelligence', link: '/guides/git-intelligence' },
    { text: 'Dependency graph', link: '/guides/dependency-graph' },
    {
      text: 'Agent configuration',
      collapsed: true,
      items: [
        { text: 'Claude Code', link: '/guides/agent-config/claude-code' },
        { text: 'Codex', link: '/guides/agent-config/codex' },
        { text: 'Cursor', link: '/guides/agent-config/cursor' },
        { text: 'Gemini CLI', link: '/guides/agent-config/geminicli' },
        { text: 'Vibe', link: '/guides/agent-config/vibe' }
      ]
    },
    {
      text: 'How-to',
      collapsed: true,
      items: [
        { text: 'Add an LLM provider', link: '/guides/howto/llm-providers' },
        { text: 'Add a file extension', link: '/guides/howto/file-types' },
        { text: 'Change the embedding model', link: '/guides/howto/embedding-model' },
        { text: 'Manage Qdrant collections', link: '/guides/howto/qdrant' }
      ]
    }
  ]
}

function getArchitectureSidebar () {
  return [
    { text: 'Overview', link: '/architecture/introduction' },
    { text: 'Ingestion job', link: '/architecture/ingestion' },
    { text: 'API', link: '/architecture/api' },
    { text: 'MCP server', link: '/architecture/mcp' }
  ]
}

function getResearchSidebar () {
  return [
    { text: 'Overview', link: '/research/introduction' },
    { text: 'nb01 · Corpus discovery', link: '/research/corpus-discovery' },
    { text: 'nb02 · Markdown chunking', link: '/research/markdown-chunking' },
    { text: 'nb03 · JS & Vue chunking', link: '/research/js-vue-chunking' },
    { text: 'nb04 · JSON chunking', link: '/research/json-chunking' },
    { text: 'nb05 · Embedding evaluation', link: '/research/embedding-evaluation' },
    { text: 'nb06 · Qdrant index & eval', link: '/research/qdrant-index-eval' }
  ]
}
