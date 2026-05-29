from api.schemas import Chunk


# Stub: return one canned chunk until the real Qdrant client is wired in
def search(vector, top_k=5):
    chunk = Chunk(
        source_path="kano/src/mixins/mixin.base-map.js",
        repository="kano",
        start_line=142,
        end_line=168,
        breadcrumb="baseMapMixin > methods > center",
        score=0.95,
        content="center (longitude, latitude, zoomLevel) {/* ... */}",
    )
    return [chunk] * top_k