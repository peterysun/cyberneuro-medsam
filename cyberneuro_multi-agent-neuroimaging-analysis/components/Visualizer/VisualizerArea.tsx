import React, { useRef, useCallback, useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { ToolVisualization, VisualizationType, GroupComparisonResult } from '../../types';
import { ScatterPlot, StatsBarChart, AgingCurveChart, ClusteringDashboard, StratificationChart, SVMBoundaryChart, ChartConfig, VisionBBoxesChart } from './Charts';
import { CFCWaveletCard, HubDetectionCard } from './Charts';
import BidsConversionCard from '../DicomProcess/BidsConversionCard';
import { HtmlVisualizationRenderer } from './HtmlVisualizationRenderer';
import DynamicChartRenderer from './DynamicChartRenderer';
import CodeEditorModal from './CodeEditorModal';
import { prepareDataScope, chartDataToCode } from '../../utils/chartToCode';
import { FileText, Database, BookOpen, Link, FileCheck2, CheckCircle2, TrendingUp, Grid2X2, Layers, Download, Binary, Pencil, Code } from 'lucide-react';
import WMBrainChart from './WMBrainChart';

interface VisualizerAreaProps {
  visualizations: ToolVisualization[];
  datasetName?: string;
  onVizClick?: (id?: string) => void;
  onHtmlChange?: (messageId: string, newHtml: string) => void;
  onCodeChange?: (vizId: string, code: string) => void;
  onConfigChange?: (vizId: string, config: ChartConfig) => void;
  activeDatasetIds?: string[];
  selectedVisualizationId?: string | null;
  isProcessing?: boolean;
}

const ResearchReport: React.FC<{ data: any, onLinkClick: (stepId: number) => void }> = ({ data, onLinkClick }) => {
  const { report, stepIdToMessageId } = data;
  const markdownContent = report.replace(/\[\[Step (\d+)\]\]/g, '[Step $1](urn:step:$1)');

  return (
    <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-700/50">
      <div className="prose prose-invert max-w-none text-slate-300 leading-relaxed">
        <ReactMarkdown
          components={{
            a: ({ node, href, children, ...props }) => {
              if (href?.startsWith('urn:step:')) {
                const stepId = parseInt(href.split(':')[2], 10);
                return (
                  <button
                    onClick={() => onLinkClick(stepId)}
                    className="text-indigo-400 hover:text-indigo-300 font-bold underline decoration-indigo-500/30 underline-offset-4 bg-indigo-500/10 px-1 rounded transition-colors inline-block"
                    title={`Go to Step ${stepId}`}
                  >
                    {children}
                  </button>
                );
              }
              return (
                <a 
                  href={href} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="text-sky-400 hover:text-sky-300 hover:underline"
                  {...props}
                >
                  {children}
                </a>
              );
            },
            h1: ({children}) => <h1 className="text-2xl font-bold text-slate-100 mb-4 pb-2 border-b border-slate-700">{children}</h1>,
            h2: ({children}) => <h2 className="text-xl font-semibold text-indigo-200 mt-6 mb-3">{children}</h2>,
            h3: ({children}) => <h3 className="text-lg font-medium text-slate-200 mt-4 mb-2">{children}</h3>,
            ul: ({children}) => <ul className="list-disc pl-5 space-y-1 mb-4 text-slate-300">{children}</ul>,
            li: ({children}) => <li className="pl-1">{children}</li>,
            strong: ({children}) => <strong className="font-semibold text-slate-100">{children}</strong>,
          }}
        >
          {markdownContent}
        </ReactMarkdown>
      </div>
    </div>
  );
};

const PairwiseTable: React.FC<{ data: GroupComparisonResult }> = ({ data }) => {
  if (!data.pairwiseComparisons || data.pairwiseComparisons.length === 0) return null;

  return (
    <div className="mt-4 overflow-x-auto">
      <h4 className="text-xs font-semibold text-slate-400 uppercase mb-2">Pairwise Comparisons</h4>
      <table className="w-full text-left text-xs text-slate-300 border-collapse">
        <thead>
          <tr className="bg-slate-900/50 border-b border-slate-700">
            <th className="px-2 py-2">Groups</th>
            <th className="px-2 py-2">Difference</th>
            <th className="px-2 py-2">Effect Size (d)</th>
            <th className="px-2 py-2">P-Value</th>
            <th className="px-2 py-2">Interpretation</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {data.pairwiseComparisons.map((comp, idx) => (
            <tr key={idx} className={comp.significant ? "bg-indigo-900/10" : ""}>
              <td className="px-2 py-2 font-medium">{comp.groupA} vs {comp.groupB}</td>
              <td className="px-2 py-2">{(comp.meanA - comp.meanB).toFixed(2)}</td>
              <td className="px-2 py-2">{comp.cohensD.toFixed(2)} ({comp.effectSize})</td>
              <td className="px-2 py-2 font-mono">{comp.pVal < 0.001 ? '<0.001' : comp.pVal.toFixed(3)}</td>
              <td className="px-2 py-2 opacity-80 max-w-[200px] truncate" title={comp.explanation}>{comp.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const VisualizationCard: React.FC<{ 
    visualization: ToolVisualization, 
    onClick?: () => void, 
    onReportLinkClick?: (stepId: number) => void,
    onHtmlChange?: (newHtml: string) => void,
    onCodeChange?: (code: string) => void,
    onConfigChange?: (config: ChartConfig) => void,
    isActiveDataset?: boolean,
    isSelected?: boolean
}> = ({ visualization, onClick, onReportLinkClick, onHtmlChange, onCodeChange, onConfigChange, isActiveDataset, isSelected }) => {
  const isClickable = true; // All cards are clickable for editing
  const bodyRef = useRef<HTMLDivElement>(null);
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [showPropertyEditor, setShowPropertyEditor] = useState(false);

  // Local editable state for chart properties
  const currentConfig = visualization.config || {};
  const [editTitle, setEditTitle] = useState(currentConfig.title || '');
  const [editXLabel, setEditXLabel] = useState(currentConfig.xAxisLabel || '');
  const [editYLabel, setEditYLabel] = useState(currentConfig.yAxisLabel || '');

  // Sync local state when visualization config changes externally
  useEffect(() => {
    const cfg = visualization.config || {};
    setEditTitle(cfg.title || '');
    setEditXLabel(cfg.xAxisLabel || '');
    setEditYLabel(cfg.yAxisLabel || '');
  }, [visualization.config]);

  // Get defaults based on chart type
  const getDefaults = () => {
    const d = visualization.data;
    switch (visualization.type) {
      case VisualizationType.SCATTER_PLOT:
        return {
          title: d.groupCol ? `Grouped Correlation: ${d.xCol} vs ${d.yCol} by ${d.groupCol}` : `Correlation: ${d.xCol} vs ${d.yCol}`,
          xLabel: d.xCol || 'X Axis',
          yLabel: d.yCol || 'Y Axis'
        };
      case VisualizationType.BOX_PLOT:
        return {
          title: `Group Comparison: ${d.valueCol} by ${d.groupCol}`,
          xLabel: d.groupCol || 'Group',
          yLabel: d.valueCol || 'Value'
        };
      case VisualizationType.AGING_CURVE:
        return {
          title: d.phenotype || 'Growth Curve',
          xLabel: 'Age (yr)',
          yLabel: 'Value'
        };
      case VisualizationType.STRATIFICATION_RESULT:
        return {
          title: `Stratification: ${d.targetCol} by ${d.groupCol}`,
          xLabel: 'Group',
          yLabel: 'Row Count'
        };
      case VisualizationType.SVM_BOUNDARY:
        return {
          title: `SVM Classification: ${d.targetCol}`,
          xLabel: d.xCol || 'X',
          yLabel: d.yCol || 'Y'
        };
      default:
        return { title: visualization.title, xLabel: 'X Axis', yLabel: 'Y Axis' };
    }
  };
  const defaults = getDefaults();

  const handlePropertySave = (field: 'title' | 'xAxisLabel' | 'yAxisLabel', value: string) => {
    if (!onConfigChange) return;
    const newConfig: ChartConfig = { ...currentConfig, [field]: value || undefined };
    // Remove empty string entries to fall back to defaults
    if (!newConfig.title) delete newConfig.title;
    if (!newConfig.xAxisLabel) delete newConfig.xAxisLabel;
    if (!newConfig.yAxisLabel) delete newConfig.yAxisLabel;
    onConfigChange(newConfig);
  };

  const hasChart = [
    VisualizationType.SCATTER_PLOT,
    VisualizationType.BOX_PLOT,
    VisualizationType.AGING_CURVE,
    VisualizationType.CFC_DASHBOARD,
    VisualizationType.HUB_DETECTION,
    VisualizationType.CLUSTERING_DASHBOARD,
    VisualizationType.STRATIFICATION_RESULT,
  ].includes(visualization.type);

  const isVisionBBoxHtml = visualization.type === VisualizationType.VIS_HTML
    && typeof visualization.data?.html === 'string'
    && (visualization.data.html.includes('data-segment-bboxes=') || visualization.data.html.includes('data-segment-bbox='));

  const handleDownloadSvg = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!bodyRef.current) return;
    const svgElement = bodyRef.current.querySelector('svg.recharts-surface');
    if (!svgElement) return;

    const clone = svgElement.cloneNode(true) as SVGSVGElement;
    const chartW = svgElement.getBoundingClientRect().width;
    const chartH = svgElement.getBoundingClientRect().height;

    // Extra space for title (top), x-label (bottom), y-label (left)
    const titleH = 30;
    const xLabelH = 24;
    const yLabelW = 24;
    const totalW = chartW + yLabelW;
    const totalH = chartH + titleH + xLabelH;

    // Build a wrapper SVG that includes labels + chart
    const wrapper = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    wrapper.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    wrapper.setAttribute('width', String(totalW));
    wrapper.setAttribute('height', String(totalH));
    wrapper.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

    // Resolve label text from config or defaults
    const cfg = visualization.config || {};
    const titleText = cfg.title || defaults.title;
    const xLabelText = cfg.xAxisLabel || defaults.xLabel;
    const yLabelText = cfg.yAxisLabel || defaults.yLabel;

    // Title text (top center)
    const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    titleEl.setAttribute('x', String(totalW / 2));
    titleEl.setAttribute('y', String(titleH * 0.7));
    titleEl.setAttribute('text-anchor', 'middle');
    titleEl.setAttribute('fill', '#e2e8f0');
    titleEl.setAttribute('font-size', '14');
    titleEl.setAttribute('font-family', 'sans-serif');
    titleEl.setAttribute('font-weight', '600');
    titleEl.textContent = titleText;
    wrapper.appendChild(titleEl);

    // Position the chart SVG offset by yLabelW (left) and titleH (top)
    clone.setAttribute('x', String(yLabelW));
    clone.setAttribute('y', String(titleH));
    clone.setAttribute('width', String(chartW));
    clone.setAttribute('height', String(chartH));
    wrapper.appendChild(clone);

    // X-axis label (bottom center)
    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', String(yLabelW + chartW / 2));
    xLabel.setAttribute('y', String(titleH + chartH + xLabelH * 0.7));
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.setAttribute('fill', '#94a3b8');
    xLabel.setAttribute('font-size', '12');
    xLabel.setAttribute('font-family', 'sans-serif');
    xLabel.textContent = xLabelText;
    wrapper.appendChild(xLabel);

    // Y-axis label (left center, rotated)
    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', String(yLabelW * 0.6));
    yLabel.setAttribute('y', String(titleH + chartH / 2));
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('fill', '#94a3b8');
    yLabel.setAttribute('font-size', '12');
    yLabel.setAttribute('font-family', 'sans-serif');
    yLabel.setAttribute('transform', `rotate(-90, ${yLabelW * 0.6}, ${titleH + chartH / 2})`);
    yLabel.textContent = yLabelText;
    wrapper.appendChild(yLabel);

    const svgData = new XMLSerializer().serializeToString(wrapper);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const fileName = `${visualization.title.replace(/[^a-zA-Z0-9]/g, '_')}.svg`;

    // Use File System Access API (system save dialog) if available, fallback otherwise
    if ('showSaveFilePicker' in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: fileName,
          types: [{
            description: 'SVG Image',
            accept: { 'image/svg+xml': ['.svg'] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      } catch (err: any) {
        // User cancelled the dialog — silently ignore
        if (err?.name === 'AbortError') return;
        console.error('Save failed:', err);
      }
    } else {

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }, [visualization.title, visualization.config, defaults]);
  
  return (
    <div 
        className={`bg-slate-800 rounded-xl border overflow-hidden shadow-xl flex-shrink-0 transition-all 
        ${isSelected ? 'border-cyan-400 ring-2 ring-cyan-400/50 shadow-cyan-900/30' : isActiveDataset ? 'border-indigo-500/60 ring-1 ring-indigo-500/30' : 'border-slate-700'}
        ${isClickable ? 'cursor-pointer hover:ring-2 hover:ring-cyan-500/40 hover:border-cyan-500/60' : ''}`}
        onClick={onClick}
    >
      {/* Header */}
      <div className="bg-slate-900 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {visualization.type === VisualizationType.SCATTER_PLOT && <Database className="w-4 h-4 text-sky-400" />}
          {visualization.type === VisualizationType.BOX_PLOT && <Database className="w-4 h-4 text-purple-400" />}
          {visualization.type === VisualizationType.AGING_CURVE && <TrendingUp className="w-4 h-4 text-teal-400" />}
          {visualization.type === VisualizationType.CFC_DASHBOARD && <TrendingUp className="w-4 h-4 text-teal-400" />}
          {visualization.type === VisualizationType.HUB_DETECTION && <TrendingUp className="w-4 h-4 text-teal-400" />}
          {visualization.type === VisualizationType.CLUSTERING_DASHBOARD && <Grid2X2 className="w-4 h-4 text-rose-400" />}
          {visualization.type === VisualizationType.STRATIFICATION_RESULT && <Layers className="w-4 h-4 text-emerald-400" />}
          {visualization.type === VisualizationType.SVM_BOUNDARY && <Binary className="w-4 h-4 text-blue-500" />}
          {visualization.type === VisualizationType.LITERATURE_LIST && <BookOpen className="w-4 h-4 text-amber-400" />}
          {visualization.type === VisualizationType.DATA_TABLE && <FileText className="w-4 h-4 text-emerald-400" />}
          {visualization.type === VisualizationType.RESEARCH_REPORT && <FileCheck2 className="w-4 h-4 text-indigo-400" />}
          {visualization.type === VisualizationType.BIDS_CONVERSION && <FileCheck2 className="w-4 h-4 text-orange-400" />}
          {visualization.type === VisualizationType.WM_BRAIN_CHART && <TrendingUp className="w-4 h-4 text-violet-400" />}
          <span className="font-semibold text-slate-200">{visualization.title}</span>
        </div>
        <div className="flex items-center gap-2">
            {isSelected && (
                <span className="text-xs px-2 py-0.5 rounded bg-cyan-600 text-white font-medium">
                  Editing
                </span>
            )}
            {isActiveDataset && (
                <span title="Active Dataset" className="flex">
                    <CheckCircle2 className="w-4 h-4 text-indigo-400" />
                </span>
            )}
            {hasChart && (
                <button
                    onClick={(e) => { e.stopPropagation(); setShowCodeEditor(true); }}
                    className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-indigo-400 transition-colors"
                    title="View / Edit Code"
                >
                    <Code className="w-3.5 h-3.5" />
                </button>
            )}
            {hasChart && (
                <button
                    onClick={(e) => { e.stopPropagation(); setShowPropertyEditor(!showPropertyEditor); }}
                    className={`p-1 rounded hover:bg-slate-700 transition-colors ${showPropertyEditor ? 'text-cyan-400 bg-slate-700' : 'text-slate-400 hover:text-cyan-400'}`}
                    title="Edit Chart Properties"
                >
                    <Pencil className="w-3.5 h-3.5" />
                </button>
            )}
            {hasChart && (
                <button
                    onClick={handleDownloadSvg}
                    className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-emerald-400 transition-colors"
                    title="Download SVG"
                >
                    <Download className="w-3.5 h-3.5" />
                </button>
            )}
            {visualization.messageId && <Link className="w-3 h-3 text-slate-500" />}
            <span className={`text-xs px-2 py-1 rounded border ${
              visualization.type === VisualizationType.VIS_HTML 
                ? 'bg-cyan-900/50 text-cyan-300 border-cyan-700' 
                : 'bg-slate-800 text-slate-400 border-slate-600'
            }`}>
            {visualization.type}
          </span>
        </div>
      </div>

      {/* Inline Property Editor Panel */}
      {hasChart && showPropertyEditor && (
        <div className="bg-slate-900/80 border-b border-slate-700 px-4 py-3 pointer-events-auto" onClick={e => e.stopPropagation()}>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">Title</label>
              <input
                type="text"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={() => handlePropertySave('title', editTitle)}
                onKeyDown={(e) => { if (e.key === 'Enter') handlePropertySave('title', editTitle); }}
                placeholder={defaults.title}
                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
              />
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">X-Axis Label</label>
              <input
                type="text"
                value={editXLabel}
                onChange={(e) => setEditXLabel(e.target.value)}
                onBlur={() => handlePropertySave('xAxisLabel', editXLabel)}
                onKeyDown={(e) => { if (e.key === 'Enter') handlePropertySave('xAxisLabel', editXLabel); }}
                placeholder={defaults.xLabel}
                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
              />
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1">Y-Axis Label</label>
              <input
                type="text"
                value={editYLabel}
                onChange={(e) => setEditYLabel(e.target.value)}
                onBlur={() => handlePropertySave('yAxisLabel', editYLabel)}
                onKeyDown={(e) => { if (e.key === 'Enter') handlePropertySave('yAxisLabel', editYLabel); }}
                placeholder={defaults.yLabel}
                className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
              />
            </div>
          </div>
          <p className="text-[10px] text-slate-600 mt-2">Press Enter or click away to apply. Clear field to reset to default.</p>
        </div>
      )}

      {/* Body */}
      <div ref={bodyRef} className="p-4 bg-slate-800/50"> 
        {/* Dynamic rendering: when customCode exists, use DynamicChartRenderer */}
        {visualization.customCode ? (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <DynamicChartRenderer
              code={visualization.customCode}
              data={prepareDataScope(visualization.type, visualization.data, visualization.config)}
            />
          </div>
        ) : (<>
        {visualization.type === VisualizationType.SCATTER_PLOT && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <ScatterPlot data={visualization.data} config={visualization.config} onConfigChange={onConfigChange} />
          </div>
        )}

        {/* Box Plot / Stats Bar Chart */}
        {visualization.type === VisualizationType.BOX_PLOT && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <StatsBarChart data={visualization.data} config={visualization.config} onConfigChange={onConfigChange} />
            <PairwiseTable data={visualization.data} />
          </div>
        )}

        {visualization.type === VisualizationType.AGING_CURVE && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <AgingCurveChart data={visualization.data} config={visualization.config} onConfigChange={onConfigChange} />
            </div>
        )}

        {visualization.type === VisualizationType.CFC_DASHBOARD && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <CFCWaveletCard data={visualization.data} timestamp={visualization.data.timestamp}/>
            </div>
        )}

        {visualization.type === VisualizationType.HUB_DETECTION && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <HubDetectionCard data={visualization.data} timestamp={visualization.data.timestamp}/>
            </div>
        )}

        {visualization.type === VisualizationType.CLUSTERING_DASHBOARD && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <ClusteringDashboard data={visualization.data} config={visualization.config} />
            </div>
        )}

        {visualization.type === VisualizationType.STRATIFICATION_RESULT && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <StratificationChart data={visualization.data} config={visualization.config} onConfigChange={onConfigChange} />
            </div>
        )}

        {visualization.type === VisualizationType.SVM_BOUNDARY && (
            <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
                <SVMBoundaryChart data={visualization.data} config={visualization.config} onConfigChange={onConfigChange} />
            </div>
        )}

        {/* Data Table */}
        {visualization.type === VisualizationType.DATA_TABLE && (
          <div className="overflow-x-auto max-h-80 custom-scrollbar pointer-events-auto" onClick={e => e.stopPropagation()}>
            <table className="w-full text-left text-sm text-slate-300">
              <thead className="bg-slate-700/50 uppercase text-xs font-semibold text-slate-400 sticky top-0">
                <tr>
                  {Object.keys(visualization.data[0] || {}).map(k => (
                    <th key={k} className="px-4 py-3 whitespace-nowrap bg-slate-700/50 backdrop-blur">{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                {visualization.data.slice(0, 10).map((row: any, i: number) => (
                  <tr key={i} className="hover:bg-slate-700/30">
                    {Object.values(row).map((val: any, j) => (
                      <td key={j} className="px-4 py-2 font-mono text-xs whitespace-nowrap">{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-slate-500 mt-2 italic">Showing first 10 rows.</p>
          </div>
        )}

        {/* Literature List */}
        {visualization.type === VisualizationType.LITERATURE_LIST && (
          <div className="space-y-4 pointer-events-auto" onClick={e => e.stopPropagation()}>
            {visualization.data.map((paper: any, idx: number) => (
              <div key={idx} className="p-4 bg-slate-900 rounded-lg border border-slate-700 transition-colors">
                <h4 className="text-md font-bold text-amber-100 mb-1">{paper.title}</h4>
                <p className="text-xs text-amber-300/80 mb-2">{paper.authors} • {paper.year} • {paper.journal}</p>
                <p className="text-sm text-slate-400 leading-relaxed">{paper.summary}</p>
              </div>
            ))}
          </div>
        )}

        {/* Research Report */}
        {visualization.type === VisualizationType.RESEARCH_REPORT && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <ResearchReport 
              data={visualization.data} 
              onLinkClick={(stepId) => onReportLinkClick && onReportLinkClick(stepId)} 
            />
          </div>
        )}

        {/* HTML Visualization (legacy) */}
        {visualization.type === VisualizationType.VIS_HTML && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            {isVisionBBoxHtml ? (
              <VisionBBoxesChart html={visualization.data?.html} />
            ) : (
              <HtmlVisualizationRenderer
                html={visualization.data?.html}
                heightPx={visualization.data?.heightPx}
                onHtmlChange={onHtmlChange}
              />
            )}
          </div>
        )}

        {visualization.type === VisualizationType.BIDS_CONVERSION && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <BidsConversionCard data={visualization.data} timestamp={visualization.data.timestamp} onComplete={visualization.data.onComplete} />
          </div>
        )}

        {visualization.type === VisualizationType.WM_BRAIN_CHART && (
          <div className="pointer-events-auto" onClick={e => e.stopPropagation()}>
            <WMBrainChart data={visualization.data} />
          </div>
        )}

        </>)}
      </div>

      {/* Code Editor Modal */}
      {hasChart && showCodeEditor && (
        <CodeEditorModal
          isOpen={showCodeEditor}
          onClose={() => setShowCodeEditor(false)}
          html={visualization.customCode || chartDataToCode(visualization.type, visualization.data, visualization.config) || '// No code available'}
          onSave={(newCode: string) => {
            if (onCodeChange) onCodeChange(newCode);
          }}
          title="Edit Recharts Code"
        />
      )}
    </div>
  );
};

const VisualizerArea: React.FC<VisualizerAreaProps> = React.memo(({ visualizations, datasetName, onVizClick, onHtmlChange, onCodeChange, onConfigChange, activeDatasetIds, selectedVisualizationId, isProcessing }) => {
  const handleReportLinkClick = (stepId: number) => {
    const reportViz = visualizations.find(v => v.type === VisualizationType.RESEARCH_REPORT);
    if (reportViz && reportViz.data.stepIdToMessageId[stepId]) {
      const messageId = reportViz.data.stepIdToMessageId[stepId];
      if (onVizClick) onVizClick(messageId);
    }
  };

  if (!visualizations || visualizations.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-500 bg-slate-900/30 rounded-xl border-2 border-dashed border-slate-700 p-8">
        <Database className="w-16 h-16 mb-4 opacity-20" />
        <p className="text-lg font-medium">Visualization Workspace</p>
        <p className="text-sm">Agent outputs will appear here. Click any chart to edit it.</p>
        {datasetName && <p className="text-xs mt-4 text-emerald-500">Loaded: {datasetName}</p>}
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-slate-950/30 rounded-xl border border-slate-800 overflow-hidden">
      {isProcessing && (
        <div className="flex items-center justify-center gap-3 py-3 border-b border-slate-800 bg-slate-900/60">
          <svg className="animate-spin h-5 w-5 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="text-sm text-slate-400">Generating visualization...</span>
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
        {visualizations.map((viz, index) => (
           <VisualizationCard 
             key={viz.vizId || index} 
             visualization={viz} 
             onClick={() => onVizClick && onVizClick(viz.vizId || viz.datasetId)}
             onReportLinkClick={handleReportLinkClick}
             onHtmlChange={onHtmlChange 
               ? (newHtml: string) => onHtmlChange(viz.vizId, newHtml) 
               : undefined
             }
             onCodeChange={onCodeChange
               ? (code: string) => onCodeChange(viz.vizId!, code)
               : undefined
             }
             onConfigChange={onConfigChange 
               ? (config: ChartConfig) => onConfigChange(viz.vizId!, config) 
               : undefined
             }
             isActiveDataset={viz.datasetId ? activeDatasetIds?.includes(viz.datasetId) : false}
             isSelected={viz.vizId === selectedVisualizationId}
           />
        ))}
      </div>
    </div>
  );
});

export default VisualizerArea;
