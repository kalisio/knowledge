import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    base: '/knowledge/',
    title: 'knowledge',
    description: 'A retrieval-augmented generation (RAG) engine specialised in the Kalisio ecosystem',
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
        { text: 'Research', link: '/research/introduction' }
      ],
      sidebar: {
        '/about/': getAboutSidebar(),
        '/guides/': getGuidesSidebar(),
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
    { text: 'Architecture', link: '/guides/architecture' },
    {
      text: 'Components',
      collapsed: false,
      items: [
        { text: 'Ingestion job', link: '/guides/components/ingestion-job' },
        { text: 'API', link: '/guides/components/api' },
        { text: 'Qdrant', link: '/guides/components/qdrant' }
      ]
    },
    {
      text: 'File formats',
      collapsed: false,
      items: [
        { text: 'Embedding & chunking', link: '/guides/file-formats/' },
        { text: 'Markdown', link: '/guides/file-formats/markdown' },
        { text: 'JavaScript', link: '/guides/file-formats/javascript' },
        { text: 'Vue', link: '/guides/file-formats/vue' },
        { text: 'JSON', link: '/guides/file-formats/json' },
        { text: 'YAML', link: '/guides/file-formats/yaml' },
        { text: 'Shell', link: '/guides/file-formats/shell' },
        { text: 'Add a file format', link: '/guides/file-formats/add-a-format' }
      ]
    }
  ]
}

function getResearchSidebar () {
  return [
    { text: 'Overview', link: '/research/introduction' },
    { text: 'Corpus discovery', link: '/research/corpus-discovery' },
    { text: 'Markdown chunking', link: '/research/markdown-chunking' },
    { text: 'JS & Vue chunking', link: '/research/js-vue-chunking' },
    { text: 'JSON chunking', link: '/research/json-chunking' },
    { text: 'Embedding model', link: '/research/embedding-evaluation' },
    { text: 'Vector store', link: '/research/qdrant-index-eval' },
    { text: 'Multilingual retrieval', link: '/research/multilingual-retrieval' }
  ]
}
