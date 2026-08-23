<template>
  <q-list dense class="layer-list">
    <q-item v-for="layer in layers" :key="layer.name" clickable>
      <q-item-section avatar>
        <q-icon :name="layer.icon" />
      </q-item-section>
      <q-item-section>{{ layer.label }}</q-item-section>
    </q-item>
  </q-list>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useCatalog } from '../composables/catalog.js'

const layers = ref([])
const { getLayers } = useCatalog()

async function refresh () {
  layers.value = await getLayers({ type: 'OverlayLayer' })
}

onMounted(refresh)

defineExpose({ refresh })
</script>

<style lang="scss" scoped>
.layer-list {
  max-height: 60vh;
  overflow-y: auto;
}
</style>
