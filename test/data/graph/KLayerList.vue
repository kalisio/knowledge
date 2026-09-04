<template>
  <q-list>
    <!-- The template holds text that looks like an import and is not one:
         import { Store } from './nowhere.js' -->
    <q-item v-for="layer in layers" :key="layer.name">{{ layer.label }}</q-item>
  </q-list>
</template>

<script>
import { Store } from '@kalisio/kdk/core.client'

export default {
  name: 'KLayerList',
  data: () => ({ layers: Store.get('layers') || [] })
}
</script>
