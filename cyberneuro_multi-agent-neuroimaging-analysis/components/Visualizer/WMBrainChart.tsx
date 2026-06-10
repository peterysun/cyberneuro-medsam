import React, { useMemo, useState, useRef, useCallback, memo } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Area, ReferenceDot
} from 'recharts';

export interface WMBrainChartData {
  csv: Record<string, number>[];
  tractMetrics: string[];
  patientAge?: number;
  patientValue?: number;
}

interface PatientResult {
  id: string;
  age: number;
  sex: string;
  diagnosis: string;
  value: number;
  centileScore?: number;
}

interface WMBrainChartProps {
  data: WMBrainChartData;
  externalPatientCsv?: string;
}

const METRICS = ['fa-mean','md-mean','ad-mean','rd-mean','volume','surface_area','avg_length'] as const;

const METRIC_LABELS: Record<string, string> = {
  'fa-mean':'FA Mean','md-mean':'MD Mean','ad-mean':'AD Mean','rd-mean':'RD Mean',
  'volume':'Volume','surface_area':'Surface Area','avg_length':'Avg Length',
};

const NORMATIVE_BACKEND_URL = 'http://localhost:8100/align';

const KNOWN_METRICS = ['fa-mean','md-mean','ad-mean','rd-mean','volume','surface_area','avg_length'];

function extractTracts(tractMetrics: string[]): string[] {
  const tracts = new Set<string>();
  tractMetrics.forEach(tm => {
    for (const metric of KNOWN_METRICS) {
      if (tm.endsWith('-' + metric)) {
        tracts.add(tm.slice(0, -(metric.length + 1)));
        return;
      }
    }
    // fallback: split on last dash
    const i = tm.lastIndexOf('-');
    if (i > 0) tracts.add(tm.substring(0, i));
  });
  return Array.from(tracts).sort();
}

function buildChartData(csv: Record<string, number>[], tract: string, metric: string, sex: 'male' | 'female') {
  const lo  = `${sex}_${tract}-${metric}_0.025_centile`;
  const mid = `${sex}_${tract}-${metric}_0.5_centile`;
  const hi  = `${sex}_${tract}-${metric}_0.975_centile`;
  return csv.map(row => ({
    age: Number(row['ages']), p2_5: Number(row[lo]), p50: Number(row[mid]), p97_5: Number(row[hi]),
    band: [Number(row[lo]), Number(row[hi])] as [number, number],
  })).filter(r => Number.isFinite(r.age) && Number.isFinite(r.p2_5) && Number.isFinite(r.p50) && Number.isFinite(r.p97_5));
}

// ── CSV auto-converter ────────────────────────────────────────────────────────
// Accepts any CSV format and tries to map columns to required fields.
// Handles: quoted fields, semicolon delimiters, alternate column names.
function parseCsvLine(line: string, delimiter: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === delimiter && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

function autoDetectDelimiter(text: string): string {
  const firstLine = text.split('\n')[0];
  const commas = (firstLine.match(/,/g) || []).length;
  const semis = (firstLine.match(/;/g) || []).length;
  const tabs = (firstLine.match(/\t/g) || []).length;
  if (tabs > commas && tabs > semis) return '\t';
  if (semis > commas) return ';';
  return ',';
}

// Maps flexible column names to canonical names
function normalizeHeaders(headers: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  const AGE_ALIASES = ['age', 'Age', 'AGE', 'age_years', 'patient_age'];
  const SEX_ALIASES = ['sex', 'Sex', 'SEX', 'gender', 'Gender', 'GENDER'];
  const DX_ALIASES  = ['diagnosis', 'Diagnosis', 'DIAGNOSIS', 'dx', 'Dx', 'DX', 'group', 'Group'];

  headers.forEach(h => {
    const hl = h.toLowerCase().trim();
    if (AGE_ALIASES.map(a => a.toLowerCase()).includes(hl)) map[h] = 'age';
    else if (SEX_ALIASES.map(a => a.toLowerCase()).includes(hl)) map[h] = 'sex';
    else if (DX_ALIASES.map(a => a.toLowerCase()).includes(hl)) map[h] = 'diagnosis';
    else map[h] = h; // keep original for tract-metric columns
  });
  return map;
}

function normalizeSex(val: string): string {
  const v = val.trim().toLowerCase();
  if (v === '1' || v === 'm' || v === 'male') return 'male';
  if (v === '0' || v === 'f' || v === 'female') return 'female';
  return 'unknown';
}

function parsePatientCsv(text: string): PatientResult[] {
  const delimiter = autoDetectDelimiter(text);
  const lines = text.trim().split('\n').filter(l => l.trim().length > 0);
  if (lines.length < 2) return [];

  const rawHeaders = parseCsvLine(lines[0], delimiter);
  const headerMap = normalizeHeaders(rawHeaders);
  const skip = new Set(['age', 'sex', 'diagnosis', 'dataset', 'subject', 'subject_id', 'id']);

  const results: PatientResult[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i], delimiter);
    if (values.length < 2) continue;

    const row: Record<string, string> = {};
    rawHeaders.forEach((h, idx) => {
      const canonical = headerMap[h];
      row[canonical] = values[idx] ?? '';
    });

    const age = parseFloat(row['age'] ?? '');
    if (!Number.isFinite(age)) continue;

    // Find the first numeric non-metadata column as the measure
    const measureCol = rawHeaders.find(h => {
      const canonical = headerMap[h];
      return !skip.has(canonical.toLowerCase()) && Number.isFinite(parseFloat(row[canonical]));
    });

    const value = measureCol ? parseFloat(row[headerMap[measureCol]]) : NaN;
    if (!Number.isFinite(value)) continue;

    results.push({
      id: `P${String(results.length + 1).padStart(3, '0')}`,
      age,
      sex: normalizeSex(row['sex'] ?? ''),
      diagnosis: row['diagnosis'] ?? 'unknown',
      value,
    });
  }

  return results;
}

function centileColor(score?: number): string {
  if (score === undefined) return '#94a3b8';
  if (score < 0.05 || score > 0.95) return '#ef4444';
  if (score < 0.10 || score > 0.90) return '#f97316';
  return '#22c55e';
}

function centileLabel(score?: number): string {
  if (score === undefined) return '—';
  return `${(score * 100).toFixed(1)}th`;
}

interface ChartPanelProps {
  chartData: { age: number; p2_5: number; p50: number; p97_5: number; band: [number, number] }[];
  yDomain: [number, number];
  metricLabel: string;
  patients: PatientResult[];
  singlePatientDot: { age: number; value: number }[];
  hasPatients: boolean;
}

const ChartPanel = memo(({ chartData, yDomain, metricLabel, patients, singlePatientDot, hasPatients }: ChartPanelProps) => (
  <div style={{ width: '100%', height: 300 }}>
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 10, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="age" type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} stroke="#475569"
          label={{ value: 'Age (years)', position: 'insideBottom', offset: -8, fill: '#64748b', fontSize: 10 }} />
        <YAxis domain={yDomain} width={64} tick={{ fontSize: 9, fill: '#64748b' }}
          tickFormatter={v => Number(v) >= 1000 ? `${(Number(v)/1000).toFixed(0)}k` : Number(v).toFixed(2)}
          stroke="#475569"
          label={{ value: metricLabel, angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10, offset: 10 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: 8, fontSize: 11, color: '#f1f5f9' }}
          labelStyle={{ color: '#a5b4fc' }}
          formatter={(v: any, name: string) => [Number(v).toFixed(2), name === 'Median (50th)' ? 'Median' : name]}
          labelFormatter={v => `Age: ${Number(v).toFixed(1)} yr`}
        />
        <Legend iconType="line" iconSize={12} verticalAlign="top" wrapperStyle={{ fontSize: 11, paddingBottom: 8 }} />
        <Area dataKey="band" stroke="none" fill="#4f46e5" fillOpacity={0.15} name="2.5–97.5th centile" legendType="none" tooltipType="none" activeDot={false} connectNulls />
        <Line dataKey="p2_5"  stroke="#475569" strokeWidth={1} strokeDasharray="4 3" dot={false} name="2.5th centile"  connectNulls />
        <Line dataKey="p97_5" stroke="#475569" strokeWidth={1} strokeDasharray="4 3" dot={false} name="97.5th centile" connectNulls />
        <Line dataKey="p50"   stroke="#a5b4fc" strokeWidth={2.5} dot={false} name="Median (50th)" connectNulls />
        {hasPatients && patients.map((p, idx) => (
          <ReferenceDot
            key={`patient-${idx}-${p.id}`}
            x={p.age}
            y={p.value}
            r={5}
            fill={centileColor(p.centileScore)}
            stroke="#1e293b"
            strokeWidth={1.5}
            ifOverflow="extendDomain"
          />
        ))}
        {singlePatientDot.length > 0 && !hasPatients && (
          <ReferenceDot
            x={singlePatientDot[0].age}
            y={singlePatientDot[0].value}
            r={6}
            fill="#f43f5e"
            stroke="#1e293b"
            strokeWidth={1.5}
            ifOverflow="extendDomain"
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  </div>
));

export const WMBrainChart: React.FC<WMBrainChartProps> = ({ data, externalPatientCsv }) => {
  const tracts = useMemo(() => extractTracts(data.tractMetrics), [data.tractMetrics]);
  const [selectedTract,  setSelectedTract]  = useState<string>(tracts[0] ?? '');
  const [selectedMetric, setSelectedMetric] = useState<string>('volume');
  const [sex,            setSex]            = useState<'male' | 'female'>('male');
  const [patientAge,     setPatientAge]     = useState<string>(data.patientAge !== undefined ? String(data.patientAge) : '');
  const [patientValue,   setPatientValue]   = useState<string>(data.patientValue !== undefined ? String(data.patientValue) : '');
  const [patients,       setPatients]       = useState<PatientResult[]>([]);
  const [csvFileName,    setCsvFileName]    = useState<string>('');
  const [isScoring,      setIsScoring]      = useState<boolean>(false);
  const [scoringError,   setScoringError]   = useState<string>('');
  const [rawCsvText,     setRawCsvText]     = useState<string>('');

  // Auto-load externally provided patient CSV from Add CSV in right panel
  React.useEffect(() => {
    if (!externalPatientCsv) return;
    setRawCsvText(externalPatientCsv);
    setCsvFileName('Uploaded via Data Context');
    setParseWarning('');

    const lines = externalPatientCsv.trim().split('\n').filter((l: string) => l.trim());
    console.log('externalPatientCsv received, preview:', externalPatientCsv.slice(0, 300));
    if (lines.length < 2) { setParseWarning('File appears empty.'); return; }

    const headers = lines[0].split(',').map((h: string) => h.trim());
    const skip = new Set(['age','sex','diagnosis','dataset','subject','subject_id','id','label_index']);

    // Find the first usable numeric column — check all rows not just first
    const measureCol = headers.find((h: string) => {
      if (skip.has(h.toLowerCase())) return false;
      const hIdx = headers.indexOf(h);
      return lines.slice(1).some((line: string) => {
        const val = line.split(',')[hIdx]?.trim();
        return val !== undefined && val !== '' && Number.isFinite(parseFloat(val));
      });
    });

    if (!measureCol) { setParseWarning('No valid numeric tract-metric columns found after conversion.'); return; }

    const rows = lines.slice(1).map((line: string, idx: number) => {
      const values = line.split(',').map((v: string) => v.trim());
      const row: Record<string, string> = {};
      headers.forEach((h: string, i: number) => { row[h] = values[i] ?? ''; });
      const ageVal = parseFloat(row['age'] ?? '');
      return {
        id: `P${String(idx + 1).padStart(3, '0')}`,
        age: Number.isFinite(ageVal) ? ageVal : 0,
        sex: row['sex'] === '1' ? 'male' : (row['sex'] === '0' ? 'female' : 'unknown'),
        diagnosis: row['diagnosis'] || 'CN',
        value: parseFloat(row[measureCol]),
        centileScore: undefined as number | undefined,
      };
    }).filter((p: any) => Number.isFinite(p.value));

    if (rows.length === 0) { setParseWarning('No valid rows found after conversion.'); return; }

    setPatients(rows);

    // Warn if age is missing
    const missingAge = rows.every((p: any) => p.age === 0);
    if (missingAge) {
      setParseWarning(`${rows.length} subject(s) loaded. Age not found in file — enter age manually below to plot on chart.`);
    }
  }, [externalPatientCsv]);
  const [parseWarning,   setParseWarning]   = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const chartData = useMemo(() => buildChartData(data.csv, selectedTract, selectedMetric, sex), [data.csv, selectedTract, selectedMetric, sex]);

  // yDomain expands to include patient dots so they're never clipped
  const yDomain = useMemo((): [number, number] => {
    if (chartData.length === 0) return [0, 1];
    let min = Infinity, max = -Infinity;
    chartData.forEach(d => { if (d.p2_5 < min) min = d.p2_5; if (d.p97_5 > max) max = d.p97_5; });
    // Expand to include any patient values outside the normative band
    patients.forEach(p => {
      if (Number.isFinite(p.value)) {
        if (p.value < min) min = p.value;
        if (p.value > max) max = p.value;
      }
    });
    const pad = (max - min) * 0.12;
    return [min - pad, max + pad];
  }, [chartData, patients]);

  const singlePatientDot = useMemo(() => {
    const a = parseFloat(patientAge), v = parseFloat(patientValue);
    if (!Number.isFinite(a) || !Number.isFinite(v)) return [];
    return [{ age: a, value: v }];
  }, [patientAge, patientValue]);

  const loadPatients = useCallback((text: string, fileName: string) => {
    setRawCsvText(text);
    setScoringError('');
    setParseWarning('');
    const parsed = parsePatientCsv(text).map(p => ({ ...p, centileScore: undefined }));
    if (parsed.length === 0) {
      setParseWarning('No valid patients found. Check that your CSV has age and at least one numeric column.');
    } else {
      setParseWarning('');
    }
    setPatients(parsed);
    setCsvFileName(fileName);
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => loadPatients(ev.target?.result as string, file.name);
    reader.readAsText(file);
  }, [loadPatients]);

  const handleRunScoring = useCallback(async () => {
    if (!rawCsvText) return;
    setIsScoring(true);
    setScoringError('');
    try {
      const response = await fetch(NORMATIVE_BACKEND_URL, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_text: rawCsvText, tract: selectedTract, metric: selectedMetric }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const result = await response.json();
      if (result.scores) {
        setPatients(prev => prev.map((p, idx) => ({ ...p, centileScore: result.scores[idx]?.centile_score ?? undefined })));
      } else if (result.error) {
        setScoringError(result.error);
      }
    } catch {
      setScoringError('Normative scoring backend not yet available. Connect the frontier Docker backend at ' + NORMATIVE_BACKEND_URL + ' to enable centile scoring.');
    } finally {
      setIsScoring(false);
    }
  }, [rawCsvText, selectedTract, selectedMetric]);

  const tractMetricKey = `${selectedTract}-${selectedMetric}`;
  const hasData = chartData.length > 0;
  const hasPatients = patients.length > 0;
  const scoredPatients = patients.filter(p => p.centileScore !== undefined);
  const abnormalPatients = scoredPatients.filter(p => p.centileScore! < 0.05 || p.centileScore! > 0.95);

  return (
    <div className="w-full bg-slate-900 rounded-lg p-4 border border-slate-700 space-y-4">
      <div className="border-b border-slate-700 pb-2 flex items-center justify-between">
        <span className="font-semibold text-slate-200 text-sm">White Matter Brain Chart -- Normative Model</span>
        <span className="text-xs text-slate-500 font-mono">{tractMetricKey}</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">Tract</label>
          <select value={selectedTract} onChange={e => setSelectedTract(e.target.value)}
            className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
            {tracts.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">Metric</label>
          <select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}
            className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
            {METRICS.map(m => <option key={m} value={m}>{METRIC_LABELS[m]}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">Sex</label>
          <div className="flex rounded overflow-hidden border border-slate-600">
            {(['male', 'female'] as const).map(s => (
              <button key={s} onClick={() => setSex(s)}
                className={`flex-1 py-1.5 text-xs font-medium transition-colors ${sex === s ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-3 text-xs flex-wrap">
        <div className="flex-1 min-w-[80px] bg-slate-800 p-2 rounded">
          <div className="text-slate-500 mb-1">Age Range</div>
          <div className="font-mono text-indigo-300">
            {chartData.length > 0 ? `${chartData[0].age.toFixed(0)}–${chartData[chartData.length-1].age.toFixed(0)} yr` : '—'}
          </div>
        </div>
        {hasPatients && (<>
          <div className="flex-1 min-w-[80px] bg-slate-800 p-2 rounded">
            <div className="text-slate-500 mb-1">Patients</div>
            <div className="font-mono text-slate-200">{patients.length}</div>
          </div>
          <div className="flex-1 min-w-[80px] bg-slate-800 p-2 rounded">
            <div className="text-slate-500 mb-1">Scored</div>
            <div className="font-mono text-emerald-400">{scoredPatients.length}</div>
          </div>
          <div className="flex-1 min-w-[80px] bg-slate-800 p-2 rounded">
            <div className="text-slate-500 mb-1">Abnormal</div>
            <div className="font-mono text-rose-400">{abnormalPatients.length}</div>
          </div>
        </>)}
      </div>

      {hasData
        ? <ChartPanel chartData={chartData} yDomain={yDomain} metricLabel={METRIC_LABELS[selectedMetric] || selectedMetric}
            patients={patients} singlePatientDot={singlePatientDot} hasPatients={hasPatients} />
        : <div className="flex items-center justify-center text-slate-500 text-sm" style={{ height: 300 }}>
            No normative data available for {tractMetricKey} ({sex})
          </div>
      }

      {hasPatients && (
        <div className="flex gap-4 text-[10px] text-slate-400 flex-wrap">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500 inline-block" /> Normal (5th–95th)</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-orange-500 inline-block" /> Borderline</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /> Abnormal (&lt;5th or &gt;95th)</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-400 inline-block" /> Not yet scored</span>
        </div>
      )}

      <div className="border-t border-slate-700 pt-3 space-y-3">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">Normative Scoring — Upload Patient CSV</p>
        <div className="border-2 border-dashed border-slate-600 rounded-lg p-4 text-center cursor-pointer hover:border-indigo-500 transition-colors"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file) {
              const reader = new FileReader();
              reader.onload = ev => loadPatients(ev.target?.result as string, file.name);
              reader.readAsText(file);
            }
          }}>
          <input ref={fileInputRef} type="file" accept=".csv,.tsv,.txt" className="hidden" onChange={handleFileChange} />
          {csvFileName
            ? <p className="text-xs text-emerald-400">✅ {csvFileName} — {patients.length} patients loaded</p>
            : <p className="text-xs text-slate-500">Drop patient CSV here or click to upload</p>}
          <p className="text-[10px] text-slate-600 mt-1">
            Accepts any CSV format. Auto-detects age, sex, diagnosis columns. Supports comma, semicolon, or tab delimiters.
          </p>
        </div>

        {parseWarning && (
          <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">⚠️ {parseWarning}</div>
        )}

        {hasPatients && (
          <button onClick={handleRunScoring} disabled={isScoring}
            className={`w-full py-2 rounded text-xs font-semibold transition-colors ${isScoring ? 'bg-slate-700 text-slate-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-500 text-white'}`}>
            {isScoring ? 'Running normative scoring...' : `Run Normative Scoring (${patients.length} patients)`}
          </button>
        )}

        {scoringError && (
          <div className="bg-amber-900/30 border border-amber-700 rounded p-3 text-xs text-amber-300">⚠️ {scoringError}</div>
        )}
      </div>

      {hasPatients && (
        <div className="border-t border-slate-700 pt-3">
          <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Patient Results — {tractMetricKey}</p>
          <div className="overflow-x-auto max-h-64 rounded border border-slate-700">
            <table className="w-full text-xs text-slate-300 border-collapse">
              <thead className="bg-slate-800 sticky top-0">
                <tr>{['ID','Age','Sex','Dx','Value','Centile','Status'].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-slate-400 font-medium">{h}</th>
                ))}</tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {patients.map((p, idx) => {
                  const color = centileColor(p.centileScore);
                  const isAbnormal = p.centileScore !== undefined && (p.centileScore < 0.05 || p.centileScore > 0.95);
                  return (
                    <tr key={`row-${idx}-${p.id}`} className={isAbnormal ? 'bg-red-900/10' : 'hover:bg-slate-800/30'}>
                      <td className="px-3 py-1.5 font-mono">{p.id}</td>
                      <td className="px-3 py-1.5">{p.age.toFixed(1)}</td>
                      <td className="px-3 py-1.5">{p.sex}</td>
                      <td className="px-3 py-1.5">{p.diagnosis}</td>
                      <td className="px-3 py-1.5 font-mono">{p.value.toFixed(2)}</td>
                      <td className="px-3 py-1.5 font-mono" style={{ color }}>{centileLabel(p.centileScore)}</td>
                      <td className="px-3 py-1.5">
                        {p.centileScore === undefined
                          ? <span className="text-slate-500">Not scored</span>
                          : isAbnormal ? <span className="text-red-400 font-medium">⚠ Abnormal</span>
                          : <span className="text-green-400">✓ Normal</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">Centile scores require the frontier Docker backend. Dots shown on chart without scores until backend is connected.</p>
        </div>
      )}

      {!hasPatients && (
        <div className="border-t border-slate-700 pt-3">
          <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Single Patient Overlay (manual)</p>
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-[10px] text-slate-500 mb-1">Age (years)</label>
              <input type="number" min={0} max={100} step={0.1} value={patientAge} onChange={e => setPatientAge(e.target.value)}
                placeholder="e.g. 45.5" className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-rose-500" />
            </div>
            <div className="flex-1">
              <label className="block text-[10px] text-slate-500 mb-1">{METRIC_LABELS[selectedMetric] || 'Value'}</label>
              <input type="number" step="any" value={patientValue} onChange={e => setPatientValue(e.target.value)}
                placeholder="patient value" className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-rose-500" />
            </div>
          </div>
          {singlePatientDot.length > 0 && <p className="text-[10px] text-rose-400 mt-1">Patient plotted on chart above.</p>}
        </div>
      )}
    </div>
  );
};

export default WMBrainChart;
