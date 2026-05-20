import React, { useMemo, useState } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Area
} from 'recharts';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface WMBrainChartData {
  /** Raw CSV rows keyed by column name */
  csv: Record<string, number>[];
  /** Pre-parsed list of available tract-metric combinations e.g. "AF_left-volume" */
  tractMetrics: string[];
  /** Optional patient overlay point */
  patientAge?: number;
  patientValue?: number;
}

interface WMBrainChartProps {
  data: WMBrainChartData;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const METRICS = [
  'fa-mean', 'md-mean', 'ad-mean', 'rd-mean',
  'volume', 'surface_area', 'avg_length',
] as const;

const METRIC_LABELS: Record<string, string> = {
  'fa-mean':       'FA Mean',
  'md-mean':       'MD Mean',
  'ad-mean':       'AD Mean',
  'rd-mean':       'RD Mean',
  'volume':        'Volume',
  'surface_area':  'Surface Area',
  'avg_length':    'Avg Length',
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract sorted unique tract names from tractMetric strings like "AF_left-volume" */
function extractTracts(tractMetrics: string[]): string[] {
  const tracts = new Set<string>();
  tractMetrics.forEach(tm => {
    const dashIdx = tm.lastIndexOf('-');
    if (dashIdx > 0) tracts.add(tm.substring(0, dashIdx));
  });
  return Array.from(tracts).sort();
}

/** Build Recharts-compatible data array from CSV rows for a given tract/metric/sex */
function buildChartData(
  csv: Record<string, number>[],
  tract: string,
  metric: string,
  sex: 'male' | 'female',
): { age: number; p2_5: number; p50: number; p97_5: number; band: [number, number] }[] {
  const lo  = `${sex}_${tract}-${metric}_0.025_centile`;
  const mid = `${sex}_${tract}-${metric}_0.5_centile`;
  const hi  = `${sex}_${tract}-${metric}_0.975_centile`;

  return csv
    .map(row => ({
      age:  Number(row['ages']),
      p2_5: Number(row[lo]),
      p50:  Number(row[mid]),
      p97_5:Number(row[hi]),
      band: [Number(row[lo]), Number(row[hi])] as [number, number],
    }))
    .filter(r =>
      Number.isFinite(r.age) &&
      Number.isFinite(r.p2_5) &&
      Number.isFinite(r.p50) &&
      Number.isFinite(r.p97_5)
    );
}

// ── Component ─────────────────────────────────────────────────────────────────

export const WMBrainChart: React.FC<WMBrainChartProps> = ({ data }) => {
  const tracts = useMemo(() => extractTracts(data.tractMetrics), [data.tractMetrics]);

  const [selectedTract,  setSelectedTract]  = useState<string>(tracts[0] ?? '');
  const [selectedMetric, setSelectedMetric] = useState<string>('volume');
  const [sex,            setSex]            = useState<'male' | 'female'>('male');

  // Patient overlay inputs (local UI state)
  const [patientAge,   setPatientAge]   = useState<string>(
    data.patientAge   !== undefined ? String(data.patientAge)   : ''
  );
  const [patientValue, setPatientValue] = useState<string>(
    data.patientValue !== undefined ? String(data.patientValue) : ''
  );

  const chartData = useMemo(
    () => buildChartData(data.csv, selectedTract, selectedMetric, sex),
    [data.csv, selectedTract, selectedMetric, sex]
  );

  const yDomain = useMemo((): [number, number] => {
    if (chartData.length === 0) return [0, 1];
    let min = Infinity, max = -Infinity;
    chartData.forEach(d => {
      if (d.p2_5 < min) min = d.p2_5;
      if (d.p97_5 > max) max = d.p97_5;
    });
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }, [chartData]);

  const patientDot = useMemo(() => {
    const a = parseFloat(patientAge);
    const v = parseFloat(patientValue);
    if (!Number.isFinite(a) || !Number.isFinite(v)) return [];
    return [{ age: a, value: v }];
  }, [patientAge, patientValue]);

  const tractMetricKey = `${selectedTract}-${selectedMetric}`;
  const hasData = chartData.length > 0;

  // ── Tooltip formatter ───────────────────────────────────────────────────────
  const tooltipFormatter = (value: any, name: string) => {
    if (name === 'Patient') return [Number(value).toFixed(4), 'Patient'];
    if (name === 'Median (50th)') return [Number(value).toFixed(4), 'Median'];
    return [Number(value).toFixed(4), name];
  };

  return (
    <div className="w-full bg-slate-900 rounded-lg p-4 border border-slate-700 space-y-4">

      {/* ── Title ────────────────────────────────────────────────────────────── */}
      <div className="border-b border-slate-700 pb-2 flex items-center justify-between">
        <span className="font-semibold text-slate-200 text-sm">
          White Matter Brain Chart
        </span>
        <span className="text-xs text-slate-500 font-mono">{tractMetricKey}</span>
      </div>

      {/* ── Controls ─────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">

        {/* Tract selector */}
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
            Tract
          </label>
          <select
            value={selectedTract}
            onChange={e => setSelectedTract(e.target.value)}
            className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          >
            {tracts.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Metric selector */}
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
            Metric
          </label>
          <select
            value={selectedMetric}
            onChange={e => setSelectedMetric(e.target.value)}
            className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          >
            {METRICS.map(m => (
              <option key={m} value={m}>{METRIC_LABELS[m]}</option>
            ))}
          </select>
        </div>

        {/* Sex toggle */}
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">
            Sex
          </label>
          <div className="flex rounded overflow-hidden border border-slate-600">
            {(['male', 'female'] as const).map(s => (
              <button
                key={s}
                onClick={() => setSex(s)}
                className={`flex-1 py-1.5 text-xs font-medium transition-colors
                  ${sex === s
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Summary stats ────────────────────────────────────────────────────── */}
      <div className="flex gap-3 text-xs">
        <div className="flex-1 bg-slate-800 p-2 rounded">
          <div className="text-slate-500 mb-1">Age Range</div>
          <div className="font-mono text-indigo-300">
            {chartData.length > 0
              ? `${chartData[0].age.toFixed(0)}–${chartData[chartData.length - 1].age.toFixed(0)} yr`
              : '—'}
          </div>
        </div>
        <div className="flex-1 bg-slate-800 p-2 rounded">
          <div className="text-slate-500 mb-1">Data points</div>
          <div className="font-mono text-slate-200">{chartData.length}</div>
        </div>
        {patientDot.length > 0 && (
          <div className="flex-1 bg-slate-800 p-2 rounded">
            <div className="text-slate-500 mb-1">Patient</div>
            <div className="font-mono text-rose-400">
              Age {patientDot[0].age.toFixed(1)}, val {patientDot[0].value.toFixed(4)}
            </div>
          </div>
        )}
      </div>

      {/* ── Chart ────────────────────────────────────────────────────────────── */}
      <div className="h-80 w-full">
        {hasData ? (
          <div className="flex h-full">
            {/* Y-axis label */}
            <div className="flex items-center justify-center flex-shrink-0" style={{ width: 20 }}>
              <span
                className="text-slate-500 text-[10px]"
                style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
              >
                {METRIC_LABELS[selectedMetric] || selectedMetric}
              </span>
            </div>

            <div className="flex-1 flex flex-col min-w-0">
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="age"
                      type="number"
                      domain={[0, 100]}
                      tick={{ fontSize: 10, fill: '#64748b' }}
                      stroke="#475569"
                    />
                    <YAxis
                      domain={yDomain}
                      width={56}
                      tick={{ fontSize: 10, fill: '#64748b' }}
                      tickFormatter={v => Number(v).toFixed(2)}
                      stroke="#475569"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        borderColor: '#475569',
                        borderRadius: 8,
                        fontSize: 11,
                        color: '#f1f5f9',
                      }}
                      labelStyle={{ color: '#a5b4fc' }}
                      formatter={tooltipFormatter}
                      labelFormatter={v => `Age: ${Number(v).toFixed(1)} yr`}
                    />
                    <Legend
                      iconType="line"
                      iconSize={12}
                      verticalAlign="top"
                      wrapperStyle={{ fontSize: 11, paddingBottom: 8 }}
                    />

                    {/* Shaded band: 2.5th – 97.5th */}
                    <Area
                      dataKey="band"
                      stroke="none"
                      fill="#4f46e5"
                      fillOpacity={0.15}
                      name="2.5–97.5th centile"
                      legendType="none"
                      tooltipType="none"
                      activeDot={false}
                      connectNulls
                    />

                    {/* Outer centile lines */}
                    <Line
                      dataKey="p2_5"
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="4 3"
                      dot={false}
                      name="2.5th centile"
                      connectNulls
                    />
                    <Line
                      dataKey="p97_5"
                      stroke="#475569"
                      strokeWidth={1}
                      strokeDasharray="4 3"
                      dot={false}
                      name="97.5th centile"
                      connectNulls
                    />

                    {/* Median */}
                    <Line
                      dataKey="p50"
                      stroke="#a5b4fc"
                      strokeWidth={2.5}
                      dot={false}
                      name="Median (50th)"
                      connectNulls
                    />

                    {/* Patient dot overlay */}
                    {patientDot.length > 0 && (
                      <Scatter
                        data={patientDot}
                        dataKey="value"
                        name="Patient"
                        fill="#f43f5e"
                        shape="star"
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* X-axis label */}
              <div className="text-center py-1">
                <span className="text-slate-500 text-[10px]">Age (years)</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-slate-500 text-sm">
            No data available for {tractMetricKey} ({sex})
          </div>
        )}
      </div>

      {/* ── Patient overlay input ─────────────────────────────────────────────── */}
      <div className="border-t border-slate-700 pt-3">
        <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
          Patient overlay (optional)
        </p>
        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-[10px] text-slate-500 mb-1">Age (years)</label>
            <input
              type="number"
              min={0}
              max={100}
              step={0.1}
              value={patientAge}
              onChange={e => setPatientAge(e.target.value)}
              placeholder="e.g. 45.5"
              className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-rose-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-[10px] text-slate-500 mb-1">
              {METRIC_LABELS[selectedMetric] || 'Value'}
            </label>
            <input
              type="number"
              step="any"
              value={patientValue}
              onChange={e => setPatientValue(e.target.value)}
              placeholder="patient value"
              className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-rose-500"
            />
          </div>
        </div>
        {patientDot.length > 0 && (
          <p className="text-[10px] text-rose-400 mt-1">
            Patient plotted as ★ on chart above.
          </p>
        )}
      </div>
    </div>
  );
};

export default WMBrainChart;
