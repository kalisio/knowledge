import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/knowledge/',
  title: 'knowledge',
  description: 'A powerful real-time Geovisualizer',
  ignoreDeadLinks: true,
  head: [
    ['link', { href: 'https://cdnjs.cloudflare.com/ajax/libs/line-awesome/1.3.0/line-awesome/css/line-awesome.min.css', rel: 'stylesheet' }],
    ['link', { rel: 'icon', href: `/images/favicon.ico` }]
  ],
  themeConfig: {
    logo: '/images/knowledge-logo.png',
    domain: 'dev.kalisio.xyz',
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
      '/research/':  getResearchSidebar()
    },
    footer: {
      copyright: 'MIT Licensed | Copyright © 2017-20xx Kalisio'
    }
  },
  vite: {
    optimizeDeps: {
			include: ['keycloak-js', 'lodash'],
		},
		ssr: {
			noExternal: ['vitepress-theme-kalisio']
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
    { text: 'Getting Started', link: '/guides/getting-started' },

    {
      text: 'Agent configuration',
      collapsed: true, 
      items: [
        { text: 'Claude Code', link: '/guides/agent-config/claude-code' },
        { text: 'Codex', link: '/guides/agent-config/codex' },
        { text: 'Vibe', link: '/guides/agent-config/vibe' },
        { text: 'Cursor', link: '/guides/agent-config/cursor' },
        { text: 'Gemini CLI', link: '/guides/agent-config/geminicli' },
      ]
    },
    {
      text: 'How-to',
      collapsed: true, 
      items: [
        { text: 'Add a LLM provider', link: '/guides/howto/llm-providers' },
        { text: 'Add a file type', link: '/guides/howto/file-types' },
        { text: 'Change embedding model', link: '/guides/howto/embedding-model' }
      ]
    }
  ]
}

function getArchitectureSidebar () {
  return [
    { text: 'Architecture', link: '/architecture/introduction' }
  ]
}

function getResearchSidebar () {
  return [
    { text: 'Overview', link: '/research/introduction' },
    { text: 'Vector DB benchmark', link: '/research/vector-db' },
    { text: 'Chuncking startegies', link: '/research/chuncking' },
    { text: 'Embedding models benchmark', link: '/research/embedding-models' },
    { text: 'Agent evaluation', link: '/research/agent-eval' }
  ]
}