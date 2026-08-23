import L from 'leaflet'

const mapProto = L.extend({}, L.Map.prototype)

L.Map.include({

    /**
     * Given a pixel coordinate relative to the origin pixel, returns the
     * corresponding pixel coordinate relative to the map container.
     *
     * @param {L.Point} point pixel screen coordinates
     * @returns {L.Point} transformed pixel point
     */
    layerPointToContainerPoint: function (point) {
        if (!this._rotate) {
            return mapProto.layerPointToContainerPoint.apply(this, arguments)
        }
        return L.point(point)
            .add(this._getRotatePanePos())
            .rotateFrom(this._bearing, this._getRotatePanePos())
            .add(this._getMapPanePos())
    },

    /**
     * Converts a coordinate from the rotated pane reference system to the
     * reference system of the not rotated map pane.
     *
     * @param {L.Point} point pixel screen coordinates
     * @returns {L.Point}
     */
    rotatedPointToMapPanePoint: function (point) {
        return L.point(point)
            .rotate(this._bearing)
            ._add(this._getRotatePanePos())
    },

    /**
     * Converts a coordinate from the map pane reference system to the
     * reference system of the rotated pane.
     *
     * @param {L.Point} point pixel screen coordinates
     * @returns {L.Point}
     */
    mapPanePointToRotatedPoint: function (point) {
        return L.point(point)
            ._subtract(this._getRotatePanePos())
            .rotate(-this._bearing)
    },

    /**
     * Offset of the specified place to the current center, in pixels.
     *
     * @param {L.LatLng} latlng map coordinates
     */
    _getCenterOffset: function (latlng) {
        let centerOffset = mapProto._getCenterOffset.apply(this, arguments)
        if (this._rotate) {
            centerOffset = centerOffset.rotate(this._bearing)
        }
        return centerOffset
    },

    /**
     * Current position of the rotate pane, or the origin when there is none.
     */
    _getRotatePanePos: function () {
        return this._rotatePanePos || new L.Point(0, 0)
    },
})

export default mapProto
