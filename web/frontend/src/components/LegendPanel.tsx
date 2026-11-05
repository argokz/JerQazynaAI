import { colorAt, formatValue, gradient } from "../format";
import type { LayerInfo } from "../types";

export function LegendPanel({ layers }: { layers: LayerInfo[] }) {
  if (layers.length === 0) {
    return (
      <section className="panel">
        <h2>Легенда</h2>
        <p className="muted">Включите слой, чтобы увидеть легенду.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <h2>Легенда</h2>
      {layers.map((layer) => {
        const { legend } = layer;
        return (
          <div key={layer.id} className="legend">
            <div className="legend-title">{layer.title}</div>
            {legend.classes.length > 0 ? (
              <ul className="legend-classes">
                {legend.classes.map((c) => (
                  <li key={c.value}>
                    <span className="swatch" style={{ background: colorAt(legend, c.value) }} />
                    {c.label}
                  </li>
                ))}
              </ul>
            ) : (
              <>
                <div className="legend-bar" style={{ background: gradient(legend) }} />
                <div className="legend-range">
                  <span>{formatValue(legend.min, legend.units)}</span>
                  <span>{formatValue(legend.max, legend.units)}</span>
                </div>
              </>
            )}
            {layer.warning && <p className="warning">{layer.warning}</p>}
          </div>
        );
      })}
    </section>
  );
}
