import type { LayerInfo, SourceInfo } from "../types";

interface Props {
  layers: LayerInfo[];
  basemapAttribution: string;
}

function uniqueSources(layers: LayerInfo[]): { source: SourceInfo; layers: string[] }[] {
  const out = new Map<string, { source: SourceInfo; layers: string[] }>();
  for (const layer of layers) {
    for (const source of layer.sources) {
      const entry = out.get(source.id) ?? { source, layers: [] };
      entry.layers.push(layer.title);
      out.set(source.id, entry);
    }
  }
  return [...out.values()];
}

export function SourcesPanel({ layers, basemapAttribution }: Props) {
  return (
    <section className="panel">
      <h2>Источники и лицензии</h2>
      <p className="muted">
        Технический разбор лицензий: docs/DATA_LICENSES_2026-11-05.md (не юридическая консультация).
      </p>
      <ul className="sources">
        {uniqueSources(layers).map(({ source, layers: used }) => (
          <li key={source.id}>
            <div className="source-title">
              {source.title}
              {source.share_alike && <span className="badge">CC BY-SA</span>}
            </div>
            <div>
              <b>Лицензия:</b> {source.license}
            </div>
            <div>
              <b>Атрибуция:</b> {source.attribution}
            </div>
            {source.notice && <div className="notice">{source.notice}</div>}
            <div className="muted">Цитата: {source.citation}</div>
            <div className="muted">Слои: {used.join(", ")}</div>
          </li>
        ))}
        <li>
          <div className="source-title">Подложка</div>
          <div>{basemapAttribution}</div>
        </li>
      </ul>
    </section>
  );
}
