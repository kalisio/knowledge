import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    base: '/knowledge/',
    title: 'knowledge',
    description: 'AI developer assistant for the Kalisio ecosystem',
    ignoreDeadLinks: true,
    // Python experiment code lives under docs/experiments/ — keep it out of the docs build.
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
        { text: 'Add a file type', link: '/guides/howto/file-types' },
        { text: 'Change the embedding model', link: '/guides/howto/embedding-model' }
      ]
    }
  ]
}

function getArchitectureSidebar () {
  return [
    { text: 'Overview', link: '/architecture/introduction' },
    { text: 'Ingestion pipeline', link: '/architecture/ingestion-pipeline' },
    { text: 'RAG: indexing & retrieval', link: '/architecture/rag' },
    { text: 'API endpoints', link: '/architecture/api' },
    { text: 'MCP tools', link: '/architecture/mcp-tools' },
    { text: 'Git intelligence', link: '/architecture/git-intelligence' },
    { text: 'Dependency graph', link: '/architecture/dependency-graph' }
  ]
}

function getResearchSidebar () {
  return [
    { text: 'Overview', link: '/research/introduction' },
    { text: 'Vector DB benchmark', link: '/research/vector-db' },
    { text: 'Chunking strategies', link: '/research/chunking' },
    { text: 'Embedding models benchmark', link: '/research/embedding-models' },
    { text: 'Agent evaluation', link: '/research/agent-eval' }
  ]
}
