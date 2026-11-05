import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, fetchCell, fetchLayers } from "./api";
import { CellCard } from "./components/CellCard";
import { LayerPanel } from "./components/LayerPanel";
import { LegendPanel } from "./components/LegendPanel";
import { MapView } from "./components/MapView";
import { SourcesPanel } from "./components/SourcesPanel";
import type { CellResponse, LayerState, LayersResponse } from "./types";

const BASEMAP_URL =
  import.meta.env.VITE_BASEMAP_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const BASEMAP_ATTRIBUTION =
  import.meta.env.VITE_BASEMAP_ATTRIBUTION || "© OpenStreetMap contributors";

function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) return `Вне области задачи: ${error.message}`;
    if (error.status === 422) return `Вне сетки задачи: ${error.message}`;
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

export function App() {
  const [data, setData] = useState<LayersResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [layerState, setLayerState] = useState<Record<string, LayerState>>({});
  const [cell, setCell] = useState<CellResponse | null>(null);
  const [cellError, setCellError] = useState<string | null>(null);
  const [cellLoading, setCellLoading] = useState(false);
  const [selected, setSelected] = useState<[number, number] | null>(null);
  const cellRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchLayers(controller.signal)
      .then((response) => {
        setData(response);
        setLayerState(
          Object.fromEntries(
            response.layers.map((l) => [l.id, { visible: l.available && l.visible, opacity: l.opacity }]),
          ),
        );
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setLoadError(errorText(error));
      });
    return () => controller.abort();
  }, []);

  const onClick = useCallback((lon: number, lat: number) => {
    cellRequest.current?.abort();
    const controller = new AbortController();
    cellRequest.current = controller;
    setSelected([lon, lat]);
    setCellLoading(true);
    setCellError(null);
    fetchCell(lon, lat, controller.signal)
      .then((response) => setCell(response))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setCell(null);
        setCellError(errorText(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setCellLoading(false);
      });
  }, []);

  const closeCell = useCallback(() => {
    cellRequest.current?.abort();
    setCell(null);
    setCellError(null);
    setCellLoading(false);
    setSelected(null);
  }, []);

  const visibleLayers = useMemo(
    () => (data ? data.layers.filter((l) => layerState[l.id]?.visible) : []),
    [data, layerState],
  );

  if (loadError) {
    return (
      <div className="fatal">
        <h1>JerQazyna — веб-карта</h1>
        <p className="error">Не удалось загрузить слои: {loadError}</p>
        <p className="muted">Проверьте, что API запущен и VITE_API_URL указывает на него.</p>
      </div>
    );
  }
  if (!data) return <div className="fatal muted">Загрузка слоёв…</div>;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>JerQazyna</h1>
          <p className="muted">
            Задача {data.task_id}: исходные слои на сетке 1 км, их покрытие, источники и
            происхождение значений.
          </p>
        </header>
        <LayerPanel
          layers={data.layers}
          state={layerState}
          onChange={(id, next) => setLayerState((prev) => ({ ...prev, [id]: next }))}
        />
        <LegendPanel layers={visibleLayers} />
        <CellCard cell={cell} error={cellError} loading={cellLoading} onClose={closeCell} />
        <SourcesPanel layers={data.layers} basemapAttribution={BASEMAP_ATTRIBUTION} />
      </aside>
      <main className="map-wrap">
        <MapView
          layers={data.layers}
          state={layerState}
          bounds={data.bounds}
          basemapUrl={BASEMAP_URL}
          basemapAttribution={BASEMAP_ATTRIBUTION}
          selected={selected}
          onClick={onClick}
        />
      </main>
    </div>
  );
}
