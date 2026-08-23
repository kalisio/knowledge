import _ from 'lodash'
import logger from 'loglevel'

export const GEOLOCATION_TIMEOUT = 30000

// Read the current position from the browser geolocation API.
export async function getCurrentPosition (options = {}) {
  const timeout = _.get(options, 'timeout', GEOLOCATION_TIMEOUT)
  return new Promise((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, { timeout })
  })
}

// Convert a browser position into the longitude/latitude pair the map uses.
export function toLongitudeLatitude (position) {
  const { longitude, latitude } = position.coords
  return [longitude, latitude]
}

export class GeolocationTracker {
  constructor (map) {
    this.map = map
    this.watchId = null
  }

  start () {
    if (this.watchId !== null) return
    this.watchId = navigator.geolocation.watchPosition((position) => {
      this.map.setCenter(toLongitudeLatitude(position))
    }, (error) => {
      logger.error('geolocation failed', error)
    })
  }

  stop () {
    if (this.watchId === null) return
    navigator.geolocation.clearWatch(this.watchId)
    this.watchId = null
  }
}

export default GeolocationTracker
