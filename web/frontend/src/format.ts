import type { Legend } from "./types";

export function formatValue(value: number | null, units: string): string {
  if (value === null) return "нет данных";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  const text = value.toLocaleString("ru-RU", { maximumFractionDigits: digits });
  return units ? `${text} ${units}` : text;
}

export function shortHash(hash: string): string {
  return `${hash.slice(0, 10)}…${hash.slice(-6)}`;
}

export function gradient(legend: Legend): string {
  const stops = legend.stops.map((s) => `${s.color} ${(s.position * 100).toFixed(1)}%`);
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

// Цвет значения по опорным точкам палитры (для легенды классов).
export function colorAt(legend: Legend, value: number): string {
  const t = Math.min(1, Math.max(0, (value - legend.min) / (legend.max - legend.min)));
  let best = legend.stops[0];
  for (const stop of legend.stops) {
    if (Math.abs(stop.position - t) < Math.abs(best.position - t)) best = stop;
  }
  return best.color;
}
