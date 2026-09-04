import { Store } from './store.js'

// A module importing a sibling by relative path -- 40% of the imports of
// the corpus take this form.
export function getUser () {
  return Store.get('user')
}
