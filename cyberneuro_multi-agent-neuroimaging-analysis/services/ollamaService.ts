
import { Ollama } from 'ollama/browser';
import { McpTool, ChatMessage, AgentType } from "../types";
import { PROMPTS } from "../constants";
import { validatePlanColumns } from './internalTools';

// Parameters the frontend injects automatically — hide from LLM to avoid hallucinated values.
const FRONTEND_INJECTED_PARAMS = new Set(['dataset_id']);

function stripInjectedParams(properties: Record<string, any>): Record<string, any> {
  const filtered: Record<string, any> = {};
  for (const [k, v] of Object.entries(properties)) {
    if (!FRONTEND_INJECTED_PARAMS.has(k)) filtered[k] = v;
  }
  return filtered;
}

function stripInjectedRequired(required: string[]): string[] {
  return required.filter(r => !FRONTEND_INJECTED_PARAMS.has(r));
}

const DEFAULT_OLLAMA_HOST = 'http://localhost:11434';
const OLLAMA_HOST_STORAGE_KEY = 'neuroagent.ollamaHost';

const normalizeHttpUrl = (value: string) => value.trim().replace(/\/$/, '');

const getStoredOllamaHost = () => {
  if (typeof window === 'undefined') return DEFAULT_OLLAMA_HOST;
  const stored = window.localStorage.getItem(OLLAMA_HOST_STORAGE_KEY);
  return stored ? normalizeHttpUrl(stored) : DEFAULT_OLLAMA_HOST;
};

let ollamaHost = getStoredOllamaHost();

let generalModel = 'llama3'; 
let neuroModel = 'llama3';
const visionModel = 'dcarrascosa/medgemma-1.5-4b-it:F16';

let ollama = new Ollama({ host: ollamaHost });

export const getOllamaHost = () => ollamaHost;

export const setOllamaHost = (nextHost: string) => {
  const normalized = normalizeHttpUrl(nextHost);
  if (!/^https?:\/\//i.test(normalized)) {
    throw new Error('Ollama URL must start with http:// or https://');
  }

  ollamaHost = normalized;
  ollama = new Ollama({ host: ollamaHost });

  if (typeof window !== 'undefined') {
    window.localStorage.setItem(OLLAMA_HOST_STORAGE_KEY, ollamaHost);
  }
};

/**
 * Attempt to parse JSON from an LLM response, with repair heuristics
 * for common issues small models produce (trailing commas, markdown
 * fences, embedded think tags, truncated output, etc.).
 */
function robustJsonParse(raw: string): any {
  // 1. Strip <think>…</think> blocks (deepseek-r1 emits these)
  let cleaned = raw.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

  // 2. Strip markdown code fences (including mid-text ones)
  cleaned = cleaned.replace(/```(?:json)?\s*/gi, '').replace(/\s*```/gi, '').trim();

  // 3. Extract the outermost { … } or [ … ] block using bracket matching
  function extractJsonBlock(text: string): string {
    const firstBrace = text.indexOf('{');
    const firstBracket = text.indexOf('[');
    let start = -1;
    let open = '{', close = '}';

    if (firstBrace >= 0 && (firstBracket < 0 || firstBrace <= firstBracket)) {
      start = firstBrace; open = '{'; close = '}';
    } else if (firstBracket >= 0) {
      start = firstBracket; open = '['; close = ']';
    }

    if (start < 0) return text;

    let depth = 0;
    let inString = false;
    let escape = false;
    let end = text.length - 1;

    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (escape) { escape = false; continue; }
      if (ch === '\\') { escape = true; continue; }
      if (ch === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (ch === open) depth++;
      else if (ch === close) depth--;
      if (depth === 0) { end = i; break; }
    }

    let block = text.substring(start, end + 1);

    // If brackets are still unbalanced (truncated output), close them
    if (depth > 0) {
      // Remove any trailing incomplete key-value pair
      block = block.replace(/,\s*"[^"]*"?\s*:?\s*"?[^"{}[\]]*$/, '');
      block = block.replace(/,\s*$/, '');
      for (let d = 0; d < depth; d++) {
        block += close;
      }
    }
    return block;
  }

  cleaned = extractJsonBlock(cleaned);

  // 4. Try raw parse first
  try { return JSON.parse(cleaned); } catch (_) { /* continue */ }

  // 5. Apply repair pipeline
  let repaired = cleaned;

  // 5a. Fix trailing commas before } or ]
  repaired = repaired.replace(/,\s*([}\]])/g, '$1');

  // 5b. Fix single-quoted strings → double-quoted
  repaired = repaired.replace(/'/g, '"');

  // 5c. Strip control chars (newlines inside strings cause parse failures)
  repaired = repaired.replace(/[\x00-\x1f]+/g, ' ');

  // 5d. Fix unquoted keys:  { key: "value" } → { "key": "value" }
  repaired = repaired.replace(/([{,]\s*)([a-zA-Z_]\w*)\s*:/g, '$1"$2":');

  // 5e. Fix missing commas between key-value pairs: }" " → }," "
  repaired = repaired.replace(/}\s*"/g, '}, "');
  repaired = repaired.replace(/"\s+"/g, '", "');

  try { return JSON.parse(repaired); } catch (_) { /* continue */ }

  // 6. More aggressive: try to extract just toolCalls array for executor responses
  const toolCallsMatch = repaired.match(/"toolCalls"\s*:\s*(\[[\s\S]*?\])\s*[,}]/);
  if (toolCallsMatch) {
    try {
      const toolCalls = JSON.parse(toolCallsMatch[1]);
      // Reconstruct a minimal valid response
      return {
        confidence: 0.7,
        needs_clarification: false,
        clarification_question: null,
        toolCalls,
        thought: "Recovered from malformed JSON"
      };
    } catch (_) { /* continue */ }
  }

  // 7. Nuclear option: regex-extract tool name and parameters
  const toolMatch = repaired.match(/"tool"\s*:\s*"([^"]+)"/);
  const paramsMatch = repaired.match(/"parameters"\s*:\s*({[^}]*})/);
  if (toolMatch) {
    let params = {};
    if (paramsMatch) {
      try { params = JSON.parse(paramsMatch[1]); } catch (_) {
        // Extract individual key-value pairs
        const kvPairs = paramsMatch[1].matchAll(/"([^"]+)"\s*:\s*"([^"]+)"/g);
        for (const kv of kvPairs) {
          (params as any)[kv[1]] = kv[2];
        }
      }
    }
    console.warn('[robustJsonParse] Recovered via regex extraction:', toolMatch[1], params);
    return {
      confidence: 0.6,
      needs_clarification: false,
      clarification_question: null,
      toolCalls: [{ tool: toolMatch[1], parameters: params }],
      thought: "Recovered from badly malformed JSON via regex"
    };
  }

  // 8. Final attempt with the repaired string  
  return JSON.parse(repaired); // let this throw if still completely unparseable
}

export const checkApiKey = () => true; 

export const checkOllamaConnection = async (): Promise<boolean> => {
  try {
    await ollama.list();
    return true;
  } catch (e) {
    console.error("Ollama connection failed:", e);
    return false;
  }
};

export const getAvailableModels = async (): Promise<string[]> => {
  try {
    const response = await ollama.list();
    return response.models.map(m => m.name);
  } catch (e) {
    console.error("Failed to fetch models:", e);
    return [];
  }
};

export const setGeneralModel = (model: string) => { generalModel = model; };
export const setNeuroModel = (model: string) => { neuroModel = model; };
export const getGeneralModel = () => generalModel;
export const getNeuroModel = () => neuroModel;
export const getVisionModel = () => visionModel;

// Helper to build summarized history
export const buildConversationContext = (messages: ChatMessage[], limit: number = 8): string => {
  if (limit <= 0) return "";

  // Take last 'limit' messages to maintain context window, excluding system noise
  const recentMessages = messages
    .filter(m => m.role !== AgentType.SYSTEM && m.role !== AgentType.PLAN_VALIDATOR && m.role !== AgentType.PREPROCESSOR)
    .slice(-limit);

  if (recentMessages.length === 0) return "";

  return recentMessages.map(m => {
    // Clean up content: remove thinking process if it's too verbose
    let content = m.content;
    if (content.length > 500) content = content.substring(0, 500) + "...(truncated)";
    return `[${m.role}]: ${content}`;
  }).join('\n\n');
};

export const classifyQuery = async (query: string): Promise<'RESEARCH' | 'GENERAL' | 'VISION' | 'DATA_MANIPULATION' | 'PREPROCESSING' | 'SEGMENTATION' | 'WM_ANALYSIS'> => {
  console.log('[Orchestrator Agent] Input:', PROMPTS.ORCHESTRATOR_CLASSIFY(query));
  try {
    const response = await ollama.generate({
      model: generalModel,
      keep_alive: 300,
      prompt: PROMPTS.ORCHESTRATOR_CLASSIFY(query),
      format: 'json',
      stream: false
    });
    const json = robustJsonParse(response.response);
    const valid = ['RESEARCH', 'GENERAL', 'VISION', 'DATA_MANIPULATION', 'PREPROCESSING', 'SEGMENTATION', 'WM_ANALYSIS'] as const;
    return valid.includes(json.category) ? json.category : 'RESEARCH';
  } catch (e) {
    console.error("Orchestrator Error:", e);
    return 'RESEARCH';
  }
};

export interface VisionAgentResult {
  basicMedicalBiologicalInfo: string;
  findings: Array<{
    finding: string;
    bbox: number[];
  }>;
}

export const runVisionAgent = async (query: string, imageBytesBase64: string): Promise<VisionAgentResult> => {
  try {
    const response: any = await ollama.generate({
      model: neuroModel,
      keep_alive: 300,
      prompt: PROMPTS.VISION_AGENT(query),
      images: [imageBytesBase64],
      format: 'json',
      stream: false
    } as any);

    const parsed = robustJsonParse(response.response || '{}');
    const basicMedicalBiologicalInfo = typeof parsed.basic_medical_biological_info === 'string' && parsed.basic_medical_biological_info.trim()
      ? parsed.basic_medical_biological_info.trim()
      : 'Basic medical/biological context is uncertain from the provided image.';

    const rawFindings = Array.isArray(parsed.findings)
      ? parsed.findings
      : (typeof parsed.finding === 'string' || Array.isArray(parsed.bbox)
          ? [{ finding: parsed.finding, bbox: parsed.bbox }]
          : []);

    const findings = rawFindings
      .map((item: any) => {
        const finding = typeof item?.finding === 'string' ? item.finding.trim() : '';
        const bbox = Array.isArray(item?.bbox)
          ? item.bbox
              .slice(0, 4)
              .map((value: any) => Number(value))
              .filter((value: number) => Number.isFinite(value))
          : [];
        return {
          finding,
          bbox: bbox.length === 4 ? bbox : []
        };
      })
      .filter((item: { finding: string; bbox: number[] }) => item.finding.length > 0 || item.bbox.length === 4);

    return {
      basicMedicalBiologicalInfo,
      findings
    };
  } catch (e) {
    console.error('Vision Agent Error:', e);
    return {
      basicMedicalBiologicalInfo: `Vision Agent [${neuroModel}]: Error ${e}.`,
      findings: []
    };
  }
};

export const generateGeneralPlan = async (query: string, availableTools: McpTool[], feedback?: string, chatHistory: string = "") => {
  // Only include name and description for high-level planning
  const toolDescriptions = availableTools.map(t => 
    `- ${t.name}: ${t.description || 'No description'}`
  ).join('\n    ');

  console.log('[General Planner Agent] Input:', PROMPTS.GENERAL_PLANNER(query, toolDescriptions, feedback || "", chatHistory));
  try {
    const response = await ollama.generate({
      model: generalModel,
      prompt: PROMPTS.GENERAL_PLANNER(query, toolDescriptions, feedback || "", chatHistory),
      format: 'json',
      stream: false,
      keep_alive: 300
    });
    return robustJsonParse(response.response);
  } catch (e) {
    console.error("General Planner Error:", e);
    return {
      analysis_steps: [],
      rationale: "Failed to generate general plan."
    };
  }
};

export const generateNeuroPlan = async (query: string, dataContext: string, availableTools: McpTool[], feedback?: string, chatHistory: string = "") => {
  // Only include name and description for high-level planning
  const toolDescriptions = availableTools.map(t => 
    `- ${t.name}: ${t.description || 'No description'}`
  ).join('\n    ');
  //    - TRANSFORM_DATA: Convert categorical columns (e.g. DX, Sex) to numeric (creates {col}_numeric). Use this before Correlation if input is categorical.
  const allToolDescs = `
    [Core Data Tools]
    - DATA_INSPECT: Display data rows to the user (Visualization). Use this when the user wants to see the table or when you need to check value formats (e.g. string vs number).
    
    [Advanced/MCP Tools]
    ${toolDescriptions ? toolDescriptions : 'No external tools available.'}
  `;

  console.log('[Neuro Planner Agent] Input:', PROMPTS.NEURO_PLANNER(query, dataContext, allToolDescs, feedback || "", chatHistory));
  try {
    const response = await ollama.generate({
      model: neuroModel,
      prompt: PROMPTS.NEURO_PLANNER(query, dataContext, allToolDescs, feedback || "", chatHistory),
      format: 'json',
      stream: false,
      keep_alive: 300
    });
    return robustJsonParse(response.response);
  } catch (e) {
    console.error("Neuro Planner Error:", e);
    return {
      analysis_steps: [
        { step_id: 1, tool: "DATA_INSPECT", description: "Inspect relevant columns (Fallback)." }
      ],
      rationale: "Fallback plan due to AI service error."
    };
  }
};

export const validatePlan = async (plan: any, availableTools: McpTool[], existingColumns: string[]) => {
  const toolManifest = availableTools.map(t => ({
    name: t.name,
    description: (t.description || '').replace(/\s*\bdata(?:set)?_id\b[^.;,\n]*/gi, ''),
    parameters: stripInjectedParams(t.inputSchema.properties || {}),
    required: stripInjectedRequired(t.inputSchema.required || [])
  }));
  console.log('[Plan Validator Agent] Input:', PROMPTS.PLAN_VALIDATOR(JSON.stringify(toolManifest), JSON.stringify(plan, null, 2)));

  try {
    // 1. Ask LLM to validate tool usage and schema
    const response = await ollama.generate({
      model: generalModel,
      prompt: PROMPTS.PLAN_VALIDATOR(JSON.stringify(toolManifest), JSON.stringify(plan, null, 2)),
      format: 'json',
      stream: false,
      keep_alive: 300
    });
    
    const result = robustJsonParse(response.response);

    // 2. If LLM response requests column validation, execute the internal tool
    if (result.check_columns && Array.isArray(result.check_columns)) {
        const colValidation = validatePlanColumns(plan, existingColumns, result.check_columns);
        if (!colValidation.valid) {
            result.valid = false;
            result.errors = [...(result.errors || []), ...colValidation.errors];
            result.suggestions = (result.suggestions || "") + " Please correct the invalid column names.";
        }
    }

    return result;
  } catch (e) {
    console.error("Plan Validator Error:", e);
    return { valid: true, errors: [], suggestions: "Validation skipped due to service error." };
  }
};

export const runExecutorAgent = async (
  instruction: string, 
  columns: string[], 
  availableTools: McpTool[], 
  clarification: string = "", 
  previousResults: string = "", 
  delegator: string = "Planner", 
  serverFilename: string | null = null,
  retryError: string = "",
  toolHint: string = ""
) => {
  const toolDefinitions = availableTools.map(t => {
    // Remove dataset_id references from description so the LLM doesn't try to supply it
    const desc = (t.description || '').replace(/\s*\bdata(?:set)?_id\b[^.;,\n]*/gi, '');
    return `Tool: ${t.name}
     Description: ${desc}
     Parameters Schema: ${JSON.stringify(stripInjectedParams(t.inputSchema.properties || {}))}`;
  }).join('\n\n');

  console.log('[Executor Agent] Input:', PROMPTS.EXECUTOR_AGENT(instruction, columns.join(', '), toolDefinitions, clarification, previousResults, delegator, serverFilename || '', retryError, toolHint));
  
  const makeRequest = async () => {
    const response = await ollama.generate({
      model: generalModel,
      prompt: PROMPTS.EXECUTOR_AGENT(instruction, columns.join(', '), toolDefinitions, clarification, previousResults, delegator, serverFilename || '', retryError, toolHint),
      format: 'json',
      stream: false,
      keep_alive: 300
    });
    return robustJsonParse(response.response);
  };

  try {
    return await makeRequest();
  } catch (firstError) {
    console.warn("Executor Agent first attempt failed, retrying:", firstError);
    try {
      // Retry once — LLM output can vary between calls
      return await makeRequest();
    } catch (secondError) {
      console.error("Executor Agent Error (both attempts):", secondError);
      throw new Error("Executor Agent failed to generate tool calls.");
    }
  }
};

function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
}

export const interpretToolResult = async (instruction: string, toolName: string, toolOutput: any) => {
  // Truncate output if too large to avoid context limit (e.g. data points)
  let outputStr = JSON.stringify(toolOutput, null, 2);
  // if (outputStr.length > 2000) outputStr = outputStr.substring(0, 2000) + "...(truncated)";

  console.log('[Executor Agent] Interpreting result...');
  try {
    const response = await ollama.generate({
      model: generalModel,
      prompt: PROMPTS.EXECUTOR_INTERPRET(instruction, toolName, outputStr),
      stream: false,
      keep_alive: 300
    });
    return stripThinkTags(response.response);
  } catch (e) {
    console.error("Executor Interpretation Error:", e);
    return "Analysis complete (could not generate detailed interpretation).";
  }
};

export const generatePreprocessingMapping = async (column: string, values: string[]) => {
  console.log('[Preprocessor Agent] Input:', PROMPTS.PREPROCESSOR_MAPPING(column, values));
  try {
    const response = await ollama.generate({
      model: neuroModel,
      prompt: PROMPTS.PREPROCESSOR_MAPPING(column, values),
      format: 'json',
      keep_alive: 300,
      stream: false
    });
    return robustJsonParse(response.response);
  } catch (e) {
    console.error("Preprocessor Error:", e);
    const fallback: Record<string, number> = {};
    values.forEach((v, i) => fallback[v] = i);
    return { mapping: fallback, rationale: "Fallback: Assigned sequential integers due to service error." };
  }
};

export const generateResearchInsights = async (results: string, availableTools: McpTool[]) => {
  const toolsStr = availableTools.map(t => `- ${t.name}: ${t.description}`).join('\n');
  console.log('[Researcher Agent] Input:', PROMPTS.RESEARCHER_INSIGHTS(results, toolsStr));
  try {
    const response = await ollama.generate({
      model: neuroModel,
      prompt: PROMPTS.RESEARCHER_INSIGHTS(results, toolsStr),
      format: 'json',
      keep_alive: 300,
      stream: false
    });
    return robustJsonParse(response.response);
  } catch (e) {
    console.error("Researcher Error:", e);
    // Fallback if JSON parsing fails or model errors
    return { 
      decision: "REPORT", 
      report: "Analysis complete. (Error generating autonomous research insights)." 
    };
  }
};

export const generatePreprocessingProposal = async (query: string) => {
  console.log('[Proposal Agent] Input:', PROMPTS.PREPROCESSING_PROPOSAL(query));
  try {
    const response = await ollama.generate({
      model: neuroModel,
      keep_alive: 300,
      prompt: PROMPTS.PREPROCESSING_PROPOSAL(query),
      format: 'json',
      stream: false
    });
    return robustJsonParse(response.response);
  } catch (e) {
    console.error("Preprocessing Proposal Error:", e);
    return { proposals: [] };
  }
};

export const generateProposalReport = async (userQuery: string, analysisResults: string, researcherNotes: string) => {
  console.log('[Proposal Reporter Agent] Input:', PROMPTS.PROPOSAL_REPORTER(userQuery, analysisResults, researcherNotes));
  try {
    const response = await ollama.generate({
      model: neuroModel,
      keep_alive: 300,
      prompt: PROMPTS.PROPOSAL_REPORTER(userQuery, analysisResults, researcherNotes),
      stream: false
    });
    return stripThinkTags(response.response);
  } catch (e) {
    console.error("Proposal Reporter Error:", e);
    return "Failed to generate report.";
  }
};
