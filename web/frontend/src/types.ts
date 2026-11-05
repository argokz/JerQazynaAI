// Типы ответов API веб-карты (web/api/jerqazyna_web/schemas.py).

export interface SourceInfo {
  id: string;
  title: string;
  license: string;
  citation: string;
  attribution: string;
  notice: string;
  share_alike: boolean;
}

export interface LegendStop {
  position: number;
  color: string;
}

export interface Legend {
  palette: string;
  min: number;
  max: number;
  units: string;
  stops: LegendStop[];
  classes: { value: number; label: string }[];
}

export interface LayerFile {
  source: string;
  source_sha256: string;
  cog_sha256: string;
  provenance_manifest: string | null;
  provenance: Record<string, string>;
}

export interface LayerInfo {
  id: string;
  title: string;
  description: string;
  units: string;
  role: "feature" | "mask" | "research";
  group: string;
  warning: string;
  available: boolean;
  unavailable_reason: string;
  visible: boolean;
  opacity: number;
  tiles_url: string | null;
  coverage: number | null;
  legend: Legend;
  sources: SourceInfo[];
  file: LayerFile | null;
}

export interface LayersResponse {
  task_id: string;
  bounds: [number, number, number, number];
  layers: LayerInfo[];
}

export interface CellValue {
  id: string;
  title: string;
  units: string;
  role: LayerInfo["role"];
  warning: string;
  available: boolean;
  value: number | null;
  sources: SourceInfo[];
  file: LayerFile | null;
}

export interface CellResponse {
  task_id: string;
  lon: number;
  lat: number;
  row: number;
  col: number;
  cell_id: number;
  x: number;
  y: number;
  cell_size_m: number;
  crs: string;
  layers: CellValue[];
}

export interface LayerState {
  visible: boolean;
  opacity: number;
}
