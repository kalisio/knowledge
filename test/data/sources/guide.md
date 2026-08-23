# Layer catalog

The catalog service stores the layer descriptors an application exposes to
its users. Each descriptor is a GeoJSON-friendly document describing where
the data comes from and how it must be rendered on the map.

## Installation

Install the module and register the service in your application:

```js
import { catalog } from '@kalisio/kdk/map.api.js'

export default function () {
  const app = this
  app.configure(catalog)
}
```

## Configuration

A descriptor is made of a `name`, a `type` and a rendering section. The
renderer is picked from the `leaflet` or `cesium` key, so the very same
descriptor drives both the 2D map and the 3D globe without any change on
the application side. Descriptors are validated against the catalog schema
before they are stored, which means a malformed renderer is rejected at
creation time rather than failing silently when the layer is displayed.

### Zoom levels

Set `minZoom` and `maxZoom` to restrict the zoom range a layer is rendered
at. Outside of that range the layer stays in the catalog but is hidden.

## Troubleshooting

When a layer never shows up, check the browser console first: a missing
`type` is the most common mistake.
