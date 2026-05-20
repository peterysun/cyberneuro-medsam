export enum AgentType {
  USER = 'User',
  ORCHESTRATOR = 'Orchestrator',
  VISION = 'Vision Agent',
  PLANNER = 'Planner', 
  GENERAL_PLANNER = 'General Planner',
  NEURO_PLANNER = 'Neuro Planner',
  PLAN_VALIDATOR = 'Plan Validator',
  PREPROCESSOR = 'Preprocessor',
  EXECUTOR = 'Executor',
  RESEARCHER = 'Researcher',
  PROPOSAL_REPORTER = 'Proposal Reporter',
  DATA_MANIPULATOR = 'Data Manipulator',
  SYSTEM = 'System'
}
export interface CFCWaveletResult {
  status: string;
  data_path: string;
  window_size: number;
  step_size: number;
  num_windows: number;
  cfcs_count: number;
  cfcs: number[][][];
  avg_cfc?: number[][];
  files_cfcs?: { filename: string; cfcs: number[][][] }[];
  files_avg_cfcs?: { filename: string; avg_cfc: number[][] }[];
  elapsed_seconds: number;
  console_output: string;
  progress: { step: string; message: string }[];
}

export interface HubDetectionResult {
  status: string;
  data_path: string;
  num_windows: number;
  k: number;
  hub_num: number;
  use_group: boolean;
  results: {
    method: string;
    hub_nodes?: number[];
    results?: { graph_index: number; hub_nodes: number[] }[];
  };
  elapsed_seconds: number;
  console_output: string;
  progress: { step: string; message: string }[];
  roi_list?: { code: string; name: string }[];
  hub_roi_images?: Record<string, string>;
}



export interface ChatMessage {
  id: string;
  role: AgentType;
  content: string;
  timestamp: number;
  isThinking?: boolean;
  metadata?: any; 
}

export interface DatasetRow {
  [key: string]: string | number;
}

export interface Dataset {
  id: string;
  name: string;
  columns: string[];
  data: DatasetRow[];
  serverFilename?: string;
}

export enum VisualizationType {
  NONE = 'NONE',
  DATA_TABLE = 'DATA_TABLE',
  SCATTER_PLOT = 'SCATTER_PLOT',
  BOX_PLOT = 'BOX_PLOT',
  AGING_CURVE = 'AGING_CURVE',
  CFC_DASHBOARD = 'CFC_DASHBOARD',
  HUB_DETECTION = 'HUB_DETECTION',
  MARKDOWN_REPORT = 'MARKDOWN_REPORT',
  LITERATURE_LIST = 'LITERATURE_LIST',
  RESEARCH_REPORT = 'RESEARCH_REPORT',
  VIS_HTML = 'VIS_HTML',
  CLUSTERING_DASHBOARD = 'CLUSTERING_DASHBOARD',
  STRATIFICATION_RESULT = 'STRATIFICATION_RESULT',
  SVM_BOUNDARY = 'SVM_BOUNDARY',
  BIDS_CONVERSION = 'BIDS_CONVERSION',
  WM_BRAIN_CHART = 'WM_BRAIN_CHART'
}

export interface ToolVisualization {
  type: VisualizationType;
  title: string;
  data: any;
  config?: any;
  vizId?: string;       // unique per visualization (for selection/keying) — auto-generated if omitted
  messageId?: string;   // links back to the chat message that created it
  datasetId?: string;
  customCode?: string;  // LLM-edited Recharts JSX code — when set, rendered dynamically instead of static component
  timestamp?: string;     // ISO timestamp for when the visualization was created — used for sorting and display
}

// NEW: Type for VIS_HTML data
export interface HtmlVisualizationData {
  html: string;
  heightPx?: number;
  version?: string;
}

export interface WMBrainChartData {
  csv: Record<string, number>[];
  tractMetrics: string[];
  patientAge?: number;
  patientValue?: number;
}

export interface AgentState {
  status: 'idle' | 'planning' | 'executing' | 'researching' | 'manipulating';
  currentTask?: string;
}

export interface PreprocessingContext {
  type: 'path_collection' | 'pipeline_execution';
  originalQuery: string;
  pathFormMessageId?: string;
  sessionId?: string;
  paths?: { data_dir: string; output_dir: string; process_dir: string; sc_fc_dir: string };
}

export interface SuspendedState {
  plan: any;
  stepIndex: number;
  data: any[];
  columns: string[];
  intent: 'RESEARCH' | 'GENERAL' | 'DATA_MANIPULATION' | 'PREPROCESSING';
  originalUserQuery?: string;
  preprocessingContext?: PreprocessingContext;
}

export interface GroupComparisonResult {
  groupCol: string;
  valueCol: string;
  groups: string[];
  pVal: number; 
  stats: { group: string; mean: number; median: number; min: number; max: number }[];
  pairwiseComparisons?: {
    groupA: string;
    groupB: string;
    testName: string;
    statistic: number;
    pVal: number;
    significant: boolean;
    cohensD: number;
    effectSize: string;
    meanA: number;
    meanB: number;
    explanation: string;
  }[];
}

export interface CorrelationSeries {
  name: string;
  r: number;
  p: number;
  n: number;
  dataPoints: { x: number; y: number; id?: string }[];
}

export interface CorrelationResult {
  xCol: string;
  yCol: string;
  groupCol?: string;
  series: CorrelationSeries[];
}

export interface SVMResult {
  xCol: string;
  yCol: string;
  targetCol: string;
  accuracy: number;
  weights: { wx: number; wy: number; b: number };
  classes: any[];
  dataPoints: { x: number; y: number; classLabel: any; predicted: any }[];
  decisionBoundary: { x1: number; y1: number; x2: number; y2: number };
}

export interface GrowthCurveResult {
  status: string;
  phenotype: string;
  data: { X: number[]; centiles: number[][]; age?: number[]; values?: number[] };
  elapsed_seconds: number;
  overlayDot_color?: number[];
}

export interface ClusteringResult {
  featureCols: string[];
  targetCol?: string;
  nCluster: number;
  pcPoints: { x: number; y: number; cluster: number; target?: number; id?: string }[];
  clusterCorrelation?: CorrelationResult;
  assignments?: { originalIndex: number; cluster: number }[];
}

export interface StratificationResult {
  targetCol: string;
  groupCol: string;
  newColumns: { name: string; count: number }[];
}

export interface McpTool {
  name: string;
  description?: string;
  inputSchema: {
    type: string;
    properties?: Record<string, any>;
    required?: string[];
  };
}

export interface McpToolCallResult {
  content: {
    type: string;
    text?: string;
    resource?: any;
  }[];
  isError?: boolean;
}