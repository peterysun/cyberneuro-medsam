
import { McpTool, DatasetRow } from '../types';
import { calculateCorrelation, getGroupStats, performSpectralClustering, stratifyDataset, calculateLinearSVM } from '../utils/stats';
import { AGING_CURVE_PHENOTYPES, getAgingCurveData } from './agingCurveData';

export const INTERNAL_TOOLS: McpTool[] = [
  {
    name: 'overlay_with_aging_curve',
    description: `Overlay the active frontend dataset on a bundled normative aging curve. No upload or file path is required. Available phenotypes: ${AGING_CURVE_PHENOTYPES.join(', ')}`,
    inputSchema: {
      type: 'object',
      properties: {
        x_phenotype: { type: 'string', description: 'Phenotype name to compare against the bundled aging curve database.' },
        age_col: { type: 'string', description: 'Column in the active dataset containing age values in months.' },
        val_col: { type: 'string', description: 'Column in the active dataset containing the metric to overlay.' }
      },
      required: ['x_phenotype', 'age_col', 'val_col']
    }
  },
  {
    name: 'DATA_INSPECT',
    description: 'Return the actual data rows to the user interface. Use this when the user explicitly asks to "see" or "show" the data, or when you need to check the format of values (e.g. strings vs numbers) within a column.',
    inputSchema: {
      type: 'object',
      properties: {
        columns: { type: 'array', items: { type: 'string' }, description: 'Specific columns to retrieve (optional)' },
        column_pattern: { type: 'string', description: 'Substring to match multiple columns to inspect (e.g. "Amyloid"). matches all columns containing this string.' }
      }
    }
  },
  {
    name: 'CORRELATION_ANALYSIS',
    description: 'Calculate Pearson correlation between two numeric columns. Optionally group by a categorical column to see correlations per group.',
    inputSchema: {
      type: 'object',
      properties: {
        x_column: { type: 'string', description: 'The first numeric column (X-axis)' },
        y_column: { type: 'string', description: 'The second numeric column (Y-axis)' },
        group_column: { type: 'string', description: 'Optional categorical column to group by (e.g. DX, Sex)' }
      },
      required: ['x_column', 'y_column']
    }
  },
  // {
  //   name: 'GROUP_COMPARISON',
  //   description: 'Compare a numeric value across all groups in a categorical column. Performs pairwise T-tests and Cohen\'s d analysis for all unique pairs.',
  //   inputSchema: {
  //     type: 'object',
  //     properties: {
  //       group_column: { type: 'string', description: 'The categorical column to group by (e.g., DX, Sex)' },
  //       target_column: { type: 'string', description: 'The numeric column to analyze (e.g., Amyloid, Tau)' }
  //     },
  //     required: ['group_column', 'target_column']
  //   }
  // },
  // {
  //   name: 'MODIFY_VISUALIZATION',
  //   description: 'Update the style of the currently visible visualization. Use this to change colors, titles, or sizes.',
  //   inputSchema: {
  //     type: 'object',
  //     properties: {
  //       color: { type: 'string', description: 'Color name or hex code (e.g. "red", "#ff0000")' },
  //       title: { type: 'string', description: 'New title for the chart' },
  //       dotSize: { type: 'number', description: 'Size of dots in scatter plot (default 100)' }
  //     }
  //   }
  // },
  // {
  //   name: 'TRANSFORM_DATA',
  //   description: 'Convert a categorical column to numeric values. This creates a new column with "_numeric" suffix (e.g. DX -> DX_numeric). Use this before correlation analysis involving categorical data.',
  //   inputSchema: {
  //     type: 'object',
  //     properties: {
  //       column: { type: 'string', description: 'The categorical column to convert (e.g., DX, Sex)' }
  //     },
  //     required: ['column']
  //   }
  // },
  {
    name: 'AVERAGE_MULTIPLE_COLUMNS',
    description: 'Calculate average values across multiple columns for each row. Matches columns by name using a "condition" string (substring match). Use this to aggregate multiple metrics (e.g. "Amyloid" matches "Amyloid_Orbital", "Amyloid_Frontal").',
    inputSchema: {
      type: 'object',
      properties: {
        condition: { type: 'string', description: 'Substring to search for in column names (e.g. "Amyloid")' }
      },
      required: ['condition']
    }
  },
  {
    name: 'SPECTRAL_CLUSTERING',
    description: 'Perform spectral clustering (PCA + K-Means) on a set of feature columns defined by a pattern, and optionally analyze correlation of clusters with a target column. Plots PC1 vs PC2 colored by Cluster.',
    inputSchema: {
      type: 'object',
      properties: {
        feature_pattern: { type: 'string', description: 'Substring that is included in multiple feature columns (e.g. "CT_") to include automatically.' },
        target_column: { type: 'string', description: 'Optional numeric column to correlate with cluster IDs (e.g., IQ, MMSE)' },
        ncluster: { type: 'number', description: 'Number of clusters (default 5)' }
      },
      required: ['feature_pattern']
    }
  },
  // {
  //   name: 'STRATIFY_DATASET',
  //   description: 'Split a dataset into subsets based on unique values of a grouping column. Creates new sparse columns for each group (e.g. IQ -> IQ_Sex_F, IQ_Sex_M) and inserts them back into the dataset. Useful for visualizing distributions across groups.',
  //   inputSchema: {
  //     type: 'object',
  //     properties: {
  //       target_column: { type: 'string', description: 'The value column to split (e.g. IQ)' },
  //       group_column: { type: 'string', description: 'The categorical/numeric column to group by (e.g. Sex, Age)' },
  //       max_group_num: { type: 'number', description: 'Max number of groups to create (default 10)' }
  //     },
  //     required: ['target_column', 'group_column']
  //   }
  // },
  {
    name: 'SVM_CLASSIFICATION',
    description: 'Fit a Linear SVM classifier to predict a target categorical column based on two numeric feature columns. Preprocesses target to binary if needed. Outputs accuracy and a plot with the decision boundary.',
    inputSchema: {
      type: 'object',
      properties: {
        x_column: { type: 'string', description: 'The first numeric feature column (X-axis)' },
        y_column: { type: 'string', description: 'The second numeric feature column (Y-axis)' },
        target_column: { type: 'string', description: 'The target categorical column (Classes)' }
      },
      required: ['x_column', 'y_column', 'target_column']
    }
  }

  ,
{
  name: 'SEGMENT_ORGAN',
  description: 'Segments an organ in a medical CT or MRI scan using MedSAM. Use when the user asks to segment, outline, or delineate an organ or tumor in a scan file (.nii.gz).',
  inputSchema: {
    type: 'object',
    properties: {
      organ: {
        type: 'string',
        description: 'The organ to segment (e.g. right kidney, left kidney, liver, heart, spleen)'
      },
      scan_path: {
        type: 'string',
        description: 'Path to the NIfTI scan file (.nii.gz)'
      }
    },
    required: ['organ', 'scan_path']
  }
}


];


const fuzzyMatchColumn = (candidate: string, data: DatasetRow[]): string => {
  if (!candidate || !data.length) return candidate;
  const actualCols = Object.keys(data[0]);

  // Exact match
  if (actualCols.includes(candidate)) return candidate;

  // Normalize: lowercase, strip underscores/spaces/hyphens
  const normalize = (s: string) => s.toLowerCase().replace(/[_\s-]/g, '');
  const normCandidate = normalize(candidate);

  for (const col of actualCols) {
    if (normalize(col) === normCandidate) return col;
  }

  // Partial containment: if candidate is a substring or vice versa
  for (const col of actualCols) {
    const normCol = normalize(col);
    if (normCol.includes(normCandidate) || normCandidate.includes(normCol)) return col;
  }

  return candidate; // fallback to original
};

const validateColumns = (data: DatasetRow[], cols: string[]) => {
    if (data.length === 0) return;
    const available = new Set(Object.keys(data[0]));
    const missing = cols.filter(c => !available.has(c));
    if (missing.length > 0) {
        throw new Error(`Input validation failed: Column(s) '${missing.join(', ')}' not found in dataset. Available columns: ${Array.from(available).join(', ')}`);
    }
};

const resolveColumnSelection = (data: DatasetRow[], explicitCols?: string[], pattern?: string): string[] => {
    const selected = new Set<string>(explicitCols || []);
    if (pattern && data.length > 0) {
        const allCols = Object.keys(data[0]);
        const lowerPattern = pattern.toLowerCase();
        allCols.filter(c => c.toLowerCase().includes(lowerPattern)).forEach(c => selected.add(c));
    }
    return Array.from(selected);
};

const toFiniteNumber = (value: string | number | undefined): number | null => {
  if (value === undefined || value === null) return null;
  const parsed = typeof value === 'number' ? value : parseFloat(String(value));
  return Number.isFinite(parsed) ? parsed : null;
};

export const executeInternalTool = async (toolName: string, args: any, data: DatasetRow[]) => {
  if (toolName === 'overlay_with_aging_curve') {
    const startedAt = performance.now();
    const phenotype = args.x_phenotype;
    let ageCol = args.age_col;
    let valCol = args.val_col;

    if (!phenotype) throw new Error("Missing required parameter 'x_phenotype'.");
    if (!ageCol || !valCol) throw new Error("Parameters 'age_col' and 'val_col' are required.");

    ageCol = fuzzyMatchColumn(ageCol, data);
    valCol = fuzzyMatchColumn(valCol, data);
    validateColumns(data, [ageCol, valCol]);

    const curveData = await getAgingCurveData(phenotype);

    const overlayPoints = data
      .map(row => {
        const age = toFiniteNumber(row[ageCol]);
        const value = toFiniteNumber(row[valCol]);
        if (age === null || value === null) return null;
        return { age: age, value };
      })
      .filter((row): row is { age: number; value: number } => row !== null);

    if (overlayPoints.length === 0) {
      throw new Error(`No valid numeric rows found for '${ageCol}' and '${valCol}'.`);
    }

    return {
      status: 'success',
      phenotype,
      elapsed_seconds: (performance.now() - startedAt) / 1000,
      data: {
        ...curveData,
        age: overlayPoints.map(point => point.age),
        values: overlayPoints.map(point => point.value)
      }
    };
  }

  if (toolName === 'DATA_INSPECT') {
    const cols = resolveColumnSelection(data, args.columns, args.column_pattern);
    
    if (cols.length > 0) {
        validateColumns(data, cols);
        // Filter data to only include requested columns
        const filteredData = data.map(row => {
            const newRow: any = {};
            cols.forEach((col: string) => newRow[col] = row[col]);
            return newRow;
        });
        return { data: filteredData };
    }
    return { data };
  }

  if (toolName === 'CORRELATION_ANALYSIS') {
    let x = args.x_column || args.target_column || args.column1 || args.x;
    let y = args.y_column || args.comparison_column || args.column2 || args.y;
    const group = args.group_column || args.group;
    
    if (!x || !y) throw new Error(`Missing columns for correlation. Received parameters: ${JSON.stringify(args)}`);
    x = fuzzyMatchColumn(x, data);
    y = fuzzyMatchColumn(y, data);
    
    const colsToCheck = [x, y];
    if (group) colsToCheck.push(group);
    validateColumns(data, colsToCheck);
    
    return calculateCorrelation(data, x, y, group);
  }

  if (toolName === 'GROUP_COMPARISON') {
    let g = args.group_column || args.group || args.groupCol;
    let t = args.target_column || args.target || args.valueCol;
    
    if (!g || !t) throw new Error(`Missing columns for group comparison. Received parameters: ${JSON.stringify(args)}`);
    g = fuzzyMatchColumn(g, data);
    t = fuzzyMatchColumn(t, data);
    validateColumns(data, [g, t]);
    
    return getGroupStats(data, g, t);
  }

  if (toolName === 'MODIFY_VISUALIZATION') {
    return args;
  }

  if (toolName === 'TRANSFORM_DATA') {
    let col = args.column;
    const mapping = args.mapping;
    if (!col) throw new Error("Missing column for transformation");
    if (!mapping) throw new Error("Missing numeric mapping for transformation");
    col = fuzzyMatchColumn(col, data);
    
    validateColumns(data, [col]);

    const newColName = `${col}_numeric`;
    const transformedData = data.map(row => ({
      ...row,
      [newColName]: mapping[row[col]] !== undefined ? mapping[row[col]] : row[col]
    }));

    return { 
        success: true, 
        transformedData, 
        newColumn: newColName,
        mapping 
    };
  }

  if (toolName === 'AVERAGE_MULTIPLE_COLUMNS') {
    const condition = args.condition;
    if (!condition || typeof condition !== 'string') {
      throw new Error("Missing or invalid 'condition' parameter for averaging.");
    }
    
    if (data.length === 0) return { data };
    
    const allCols = Object.keys(data[0]);
    const matchedCols = allCols.filter(c => c.toLowerCase().includes(condition.toLowerCase()));

    if (matchedCols.length === 0) {
         throw new Error(`No columns found matching condition '${condition}'. Available: ${allCols.slice(0, 5).join(', ')}...`);
    }

    // Generate new column name
    const cleanCond = condition.replace(/[^a-zA-Z0-9]/g, '');
    const newColName = `avg_${cleanCond}`;

    const transformedData = data.map(row => {
      let sum = 0;
      let count = 0;
      matchedCols.forEach(c => {
        const val = parseFloat(String(row[c]));
        if (!isNaN(val)) {
          sum += val;
          count++;
        }
      });
      const avg = count > 0 ? parseFloat((sum / count).toFixed(4)) : 0;
      return {
        ...row,
        [newColName]: avg
      };
    });

    return {
      success: true,
      transformedData,
      newColumn: newColName,
      matchedColumns: matchedCols
    };
  }

  if (toolName === 'SPECTRAL_CLUSTERING') {
    const features = resolveColumnSelection(data, [], args.feature_pattern);
    const target = args.target_column || args.target; // Optional
    const k = args.ncluster || 5;

    if (features.length === 0) throw new Error("No feature columns found matching the pattern. Please provide a valid 'feature_pattern'.");
    
    const colsToValidate = [...features];
    if (target) colsToValidate.push(target);
    validateColumns(data, colsToValidate);

    const result = performSpectralClustering(data, features, target, k);

    // Inject 'colors' column
    const transformedData = data.map((row, idx) => ({ ...row, colors: '' })); // Initialize
    if (result.assignments) {
        result.assignments.forEach(a => {
            transformedData[a.originalIndex].colors = String(a.cluster); // Use string for consistency in DatasetRow
        });
    }

    return {
        success: true, 
        transformedData,
        newColumn: 'colors',
        ...result 
    };
  }

  if (toolName === 'STRATIFY_DATASET') {
      let target = args.target_column || args.target;
      let group = args.group_column || args.group;
      const max = args.max_group_num || 10;
      if (target) target = fuzzyMatchColumn(target, data);
      if (group) group = fuzzyMatchColumn(group, data);
      
      if (!target || !group) throw new Error("Missing columns for stratification.");
      validateColumns(data, [target, group]);

      const { transformedData, result } = stratifyDataset(data, target, group, max);
      
      // Extract new column names for return info
      const newColNames = result.newColumns.map(c => c.name);
      
      return {
          success: true,
          transformedData,
          result, // StratificationResult
          newColumns: newColNames // For App.tsx to update active cols
      };
  }

  if (toolName === 'SVM_CLASSIFICATION') {
      const x = args.x_column;
      const y = args.y_column;
      const target = args.target_column;
      
      if (!x || !y || !target) throw new Error("Missing columns for SVM Classification.");
      validateColumns(data, [x, y, target]);

      return calculateLinearSVM(data, x, y, target);
  }

  if (toolName === 'SEGMENT_ORGAN') {
  const organ = args.organ;
  const scanPath = args.scan_path;

  if (!organ) throw new Error("Missing required parameter 'organ'.");
  if (!scanPath) throw new Error("Missing required parameter 'scan_path'.");

  const response = await fetch('http://localhost:8099/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      organ: organ,
      scan_path: scanPath
    })
  });

  if (!response.ok) {
    throw new Error(`MedSAM server error: ${response.status} ${response.statusText}`);
  }

  const result = await response.json();

  if (result.status === 'error') {
    throw new Error(`MedSAM segmentation failed: ${result.error}`);
  }

  return result;
}

  throw new Error(`Tool ${toolName} not found internally.`);
};

// Programmatically validate column references in the plan
export const validatePlanColumns = (plan: any, initialColumns: string[], columnsToCheck?: string[][]) => {
  const knownColumns = new Set(initialColumns);
  const errors: string[] = [];

  if (!plan.analysis_steps || !Array.isArray(plan.analysis_steps)) {
    return { valid: false, errors: ["Invalid plan format"] };
  }

  plan.analysis_steps.forEach((step: any, index: number) => {
    // Check if specific columns are requested for validation for this step by the Planner/Validator
    if (columnsToCheck && Array.isArray(columnsToCheck) && Array.isArray(columnsToCheck[index])) {
        const cols = columnsToCheck[index];
        cols.forEach(col => {
            if (col && typeof col === 'string') {
                 if (!knownColumns.has(col)) {
                    errors.push(`Step ${step.step_id} (${step.tool}): Column '${col}' not found in dataset (and not created by previous steps).`);
                 }
            }
        });
    }

    // Always track column creation for subsequent steps
    const params = step.parameters || {};
    if (step.tool === 'TRANSFORM_DATA' && params.column) {
      knownColumns.add(`${params.column}_numeric`);
    }
    // Track new columns from averaging, though we don't know the name deterministically here without the timestamp/randomness. 
    // In a rigorous validator, we might need to predict the name or use a fixed naming schema.
    // For now, we won't strictly validate the *existence* of the future averaged column name in subsequent steps 
    // because the exact name is generated at runtime (avg_N_cols_timestamp).
    // The planner should ideally instruct to use "the new averaged column".
  });

  return { valid: errors.length === 0, errors };
};
