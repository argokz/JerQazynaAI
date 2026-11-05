import type { LayerInfo, LayerState } from "../types";

interface Props {
  layers: LayerInfo[];
  state: Record<string, LayerState>;
  onChange: (id: string, next: LayerState) => void;
}

function groups(layers: LayerInfo[]): [string, LayerInfo[]][] {
  const out = new Map<string, LayerInfo[]>();
  for (const layer of layers) {
    const key = layer.group || "Прочее";
    out.set(key, [...(out.get(key) ?? []), layer]);
  }
  return [...out.entries()];
}

export function LayerPanel({ layers, state, onChange }: Props) {
  return (
    <section className="panel">
      <h2>Слои</h2>
      {groups(layers).map(([group, items]) => (
        <div key={group} className="layer-group">
          <h3>{group}</h3>
          {items.map((layer) => {
            const current = state[layer.id];
            return (
              <div key={layer.id} className={`layer ${layer.available ? "" : "layer-off"}`}>
                <label className="layer-title">
                  <input
                    type="checkbox"
                    checked={current?.visible ?? false}
                    disabled={!layer.available}
                    onChange={(e) => onChange(layer.id, { ...current, visible: e.target.checked })}
                  />
                  <span>{layer.title}</span>
                  {layer.role === "research" && <span className="badge badge-warn">исследование</span>}
                  {layer.sources.some((s) => s.share_alike) && (
                    <span className="badge">CC BY-SA</span>
                  )}
                </label>
                {layer.description && <p className="layer-desc">{layer.description}</p>}
                {layer.warning && <p className="warning">{layer.warning}</p>}
                {!layer.available && <p className="muted">Недоступен: {layer.unavailable_reason}</p>}
                {layer.available && (
                  <div className="layer-meta">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={current?.opacity ?? layer.opacity}
                      aria-label={`Непрозрачность: ${layer.title}`}
                      onChange={(e) =>
                        onChange(layer.id, { ...current, opacity: Number(e.target.value) })
                      }
                    />
                    {layer.coverage !== null && (
                      <span className="muted" title="Доля ячеек области задачи с данными">
                        покрытие {(layer.coverage * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </section>
  );
}
