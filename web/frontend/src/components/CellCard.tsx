import { formatValue, shortHash } from "../format";
import type { CellResponse } from "../types";

interface Props {
  cell: CellResponse | null;
  error: string | null;
  loading: boolean;
  onClose: () => void;
}

export function CellCard({ cell, error, loading, onClose }: Props) {
  if (!cell && !error && !loading) {
    return (
      <section className="panel">
        <h2>Карточка ячейки</h2>
        <p className="muted">Щёлкните по карте, чтобы увидеть значения слоёв и их происхождение.</p>
      </section>
    );
  }
  return (
    <section className="panel cell-card">
      <div className="panel-head">
        <h2>Карточка ячейки</h2>
        <button type="button" className="link" onClick={onClose}>
          закрыть
        </button>
      </div>
      {loading && <p className="muted">Загрузка…</p>}
      {error && <p className="error">{error}</p>}
      {cell && !loading && (
        <>
          <p className="cell-head">
            Ячейка {cell.row}, {cell.col} (id {cell.cell_id}), {cell.cell_size_m / 1000} км;{" "}
            {cell.lat.toFixed(4)}° с.ш., {cell.lon.toFixed(4)}° в.д.
          </p>
          <table className="cell-table">
            <tbody>
              {cell.layers.map((layer) => (
                <tr key={layer.id}>
                  <td>
                    {layer.title}
                    {layer.role === "research" && <span className="badge badge-warn">иссл.</span>}
                  </td>
                  <td className="value">
                    {layer.available ? formatValue(layer.value, layer.units) : "не экспортирован"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Происхождение</h3>
          {cell.layers.map((layer) => {
            const file = layer.file;
            if (!layer.available || !file) return null;
            return (
              <details key={layer.id} className="provenance">
                <summary>{layer.title}</summary>
                {layer.warning && <p className="warning">{layer.warning}</p>}
                <dl>
                  <dt>Источники</dt>
                  <dd>
                    {layer.sources.map((s) => (
                      <div key={s.id}>
                        {s.title} — <i>{s.license}</i>
                      </div>
                    ))}
                  </dd>
                  <dt>Файл</dt>
                  <dd className="mono">{file.source}</dd>
                  <dt>sha256 файла</dt>
                  <dd className="mono" title={file.source_sha256}>
                    {shortHash(file.source_sha256)}
                  </dd>
                  <dt>sha256 COG</dt>
                  <dd className="mono" title={file.cog_sha256}>
                    {shortHash(file.cog_sha256)}
                  </dd>
                  {file.provenance_manifest && (
                    <>
                      <dt>Подтверждено</dt>
                      <dd className="mono">
                        {file.provenance_manifest}
                        {Object.entries(file.provenance).map(([k, v]) => (
                          <div key={k} title={v}>
                            {k}: {v.length > 24 ? shortHash(v) : v}
                          </div>
                        ))}
                      </dd>
                    </>
                  )}
                </dl>
              </details>
            );
          })}
        </>
      )}
    </section>
  );
}
