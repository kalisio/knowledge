// The shape a KDK store takes: a singleton object other modules read from.
export const Store = {
  values: {},

  get (path) {
    return this.values[path]
  },

  set (path, value) {
    this.values[path] = value
  }
}
