import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    base: '/knowledge/',
    title: 'knowledge',
    description: 'AI developer assistant for the Kalisio ecosystem',
    ignoreDeadLinks: true,
    // docs/experiments/*.md are the experiment write-ups (rendered). The Python code,
    // notebooks (.ipynb) and the old project README in that tree are not docs pages —
    // VitePress only builds .md, so we only need to keep the stray old README out.
    srcExclude: ['experiments/README.md'],
    head: [
      ['link', { href: 'https://cdnjs.cloudflare.com/ajax/libs/line-awesome/1.3.0/line-awesome/css/line-awesome.min.css', rel: 'stylesheet' }],
      ['link', { rel: 'icon', href: `/images/favicon.ico` }]
    ],
    themeConfig: {
      logo: '/images/knowledge-logo.png',
      socialLinks: [{ icon: 'github', link: 'https://github.com/kalisio/knowledge' }],
      nav: [
        { text: 'About', link: '/about/introduction' },
        { text: 'Architecture', link: '/architecture/introduction' },
        { text: 'Research', link: '/research/introduction' },
        { text: 'Experiments', link: '/experiments/introduction' }
      ],
      sidebar: {
        '/about/': getAboutSidebar(),
        '/architecture/': getArchitectureSidebar(),
        '/research/': getResearchSidebar(),
        '/experiments/': getExperimentsSidebar()
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

function getArchitectureSidebar () {
  return [
    { text: 'Overview', link: '/architecture/introduction' },
    { text: 'Ingestion pipeline', link: '/architecture/ingestion-pipeline' },
    { text: 'RAG: indexing & retrieval', link: '/architecture/rag' },
    { text: 'API endpoints', link: '/architecture/api' }
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

function getExperimentsSidebar () {
  return [
    { text: 'Overview', link: '/experiments/introduction' },
    { text: 'nb01 · Corpus discovery', link: '/experiments/corpus-discovery' },
    { text: 'nb02 · Markdown chunking', link: '/experiments/markdown-chunking' },
    { text: 'nb03 · JS & Vue chunking', link: '/experiments/js-vue-chunking' },
    { text: 'nb04 · JSON chunking', link: '/experiments/json-chunking' },
    { text: 'nb05 · Embedding evaluation', link: '/experiments/embedding-evaluation' },
    { text: 'nb06 · Qdrant index & eval', link: '/experiments/qdrant-index-eval' }
  ]
}
