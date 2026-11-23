import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/knowledge/',
  title: 'knowledge',
  description: 'A retrieval-augmented generation (RAG) engine specialised in the Kalisio ecosystem',
  ignoreDeadLinks: true,
  head: [
    ['link', { href: 'https://cdnjs.cloudflare.com/ajax/libs/line-awesome/1.3.0/line-awesome/css/line-awesome.min.css', rel: 'stylesheet' }],
    ['link', { rel: 'icon', href: `/images/favicon.ico` }]
  ],
  themeConfig: {
    logo: '/images/knowledge-logo.png',
    socialLinks: [{ icon: 'github', link: 'https://github.com/kalisio/knowledge' }],
    nav: [
      { text: 'About', link: '/about/introduction' },
      { text: 'Guides', link: '/guides/understanding' }
    ],
    sidebar: {
      '/about/': getAboutSidebar(),
      '/guides/': getGuidesSidebar()
    },
    footer: {
      copyright: 'MIT Licensed | Copyright © 2017-20xx Kalisio'
    }
  },
  vite: {
    ssr: {
      noExternal: ['@kalisio/vitepress-theme']
    }
  }
})

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
        { text: 'Qdrant', link: '/guides/components/qdrant' },
        { text: 'Multilingual retrieval', link: '/guides/components/multilingual-retrieval' }
      ]
    },
    {
      text: 'File formats',
      collapsed: false,
      items: [
        { text: 'Overview', link: '/guides/file-formats/' },
        { text: 'Markdown', link: '/guides/file-formats/markdown' },
        { text: 'JavaScript', link: '/guides/file-formats/javascript' },
        { text: 'Vue', link: '/guides/file-formats/vue' },
        { text: 'JSON', link: '/guides/file-formats/json' },
        { text: 'YAML', link: '/guides/file-formats/yaml' },
        { text: 'Shell', link: '/guides/file-formats/shell' }
      ]
    }
  ]
}

