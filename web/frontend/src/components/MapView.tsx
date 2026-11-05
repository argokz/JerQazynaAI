import maplibregl, {
  type GeoJSONSource,
  type Map as MapLibreMap,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection } from "geojson";
import { useEffect, useRef } from "react";
import type { LayerInfo, LayerState } from "../types";

// Растры 1 км: дальше z10 тайлы только растягиваются, лишние запросы к TiTiler не нужны.
const MAX_SOURCE_ZOOM = 10;
const SELECTED = "selected-cell";

interface Props {
  layers: LayerInfo[];
  state: Record<string, LayerState>;
  bounds: [number, number, number, number];
  basemapUrl: string;
  basemapAttribution: string;
  selected: [number, number] | null;
  onClick: (lon: number, lat: number) => void;
}

function baseStyle(url: string, attribution: string): StyleSpecification {
  return {
    version: 8,
    sources: {
      basemap: { type: "raster", tiles: [url], tileSize: 256, attribution, maxzoom: 19 },
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap" }],
  };
}

function selectedData(point: [number, number] | null): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: point
      ? [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: point } }]
      : [],
  };
}

function applyLayerState(map: MapLibreMap, state: Record<string, LayerState>): void {
  for (const [id, s] of Object.entries(state)) {
    if (!map.getLayer(id)) continue;
    map.setLayoutProperty(id, "visibility", s.visible ? "visible" : "none");
    map.setPaintProperty(id, "raster-opacity", s.opacity);
  }
}

function applySelected(map: MapLibreMap, point: [number, number] | null): void {
  const source = map.getSource<GeoJSONSource>(SELECTED);
  source?.setData(selectedData(point));
}

function attributionOf(layer: LayerInfo): string {
  return layer.sources.map((s) => s.attribution).join("; ");
}

export function MapView(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const loaded = useRef(false);
  // Последние значения props для обработчиков карты, созданных один раз.
  const latest = useRef(props);
  latest.current = props;

  // Карта создаётся один раз: слои известны из /api/layers до монтирования.
  // Изменения видимости, прозрачности и выбранной точки применяются эффектами ниже;
  // до события load их применит сам обработчик load из latest.
  useEffect(() => {
    if (!container.current) return;
    const initial = latest.current;
    const map = new maplibregl.Map({
      container: container.current,
      style: baseStyle(initial.basemapUrl, initial.basemapAttribution),
      bounds: initial.bounds,
      fitBoundsOptions: { padding: 20 },
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("load", () => {
      for (const layer of initial.layers) {
        if (!layer.available || !layer.tiles_url) continue;
        map.addSource(layer.id, {
          type: "raster",
          tiles: [layer.tiles_url],
          tileSize: 256,
          bounds: initial.bounds,
          maxzoom: MAX_SOURCE_ZOOM,
          attribution: attributionOf(layer),
        });
        map.addLayer({
          id: layer.id,
          type: "raster",
          source: layer.id,
          layout: { visibility: "none" },
          paint: { "raster-opacity": layer.opacity, "raster-resampling": "nearest" },
        });
      }
      map.addSource(SELECTED, { type: "geojson", data: selectedData(null) });
      map.addLayer({
        id: SELECTED,
        type: "circle",
        source: SELECTED,
        paint: {
          "circle-radius": 6,
          "circle-color": "#ffffff",
          "circle-stroke-color": "#111111",
          "circle-stroke-width": 2,
        },
      });
      loaded.current = true;
      applyLayerState(map, latest.current.state);
      applySelected(map, latest.current.selected);
    });
    map.on("click", (e) => latest.current.onClick(e.lngLat.lng, e.lngLat.lat));
    mapRef.current = map;
    return () => {
      loaded.current = false;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (mapRef.current && loaded.current) applyLayerState(mapRef.current, props.state);
  }, [props.state]);

  useEffect(() => {
    if (mapRef.current && loaded.current) applySelected(mapRef.current, props.selected);
  }, [props.selected]);

  return <div ref={container} className="map" />;
}
