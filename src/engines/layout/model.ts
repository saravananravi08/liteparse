/**
 * Model download and cache management for layout detection.
 *
 * Downloads pre-converted ONNX models on first use and caches
 * them in ~/.liteparse/models/.
 */

import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import os from "os";
import axios from "axios";

/** Default model to use for layout detection. */
export const DEFAULT_MODEL = "picodet_s_layout_17cls";

/** Model registry with download URLs and metadata. */
const MODEL_REGISTRY: Record<
  string,
  {
    url: string;
    filename: string;
    sizeBytes: number;
    numClasses: number;
    inputSize: [number, number];
    labels: string[];
    hasPostProcess: boolean;
  }
> = {
  picodet_s_layout_17cls: {
    url: "", // Must be pre-converted from HuggingFace: PaddlePaddle/PicoDet-S_layout_17cls
    filename: "model.onnx",
    sizeBytes: 4_800_000,
    numClasses: 17,
    inputSize: [480, 480],
    labels: [
      "paragraph_title", "image", "text", "number", "abstract",
      "content", "figure_title", "formula", "table", "table_title",
      "reference", "doc_title", "footnote", "header", "algorithm",
      "footer", "seal",
    ],
    hasPostProcess: true,
  },
  picodet_lcnet_x1_0_fgd_layout_infer: {
    url: "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
    filename: "model.onnx",
    sizeBytes: 7_400_000,
    numClasses: 5,
    inputSize: [800, 608],
    labels: ["text", "title", "list", "table", "figure"],
    hasPostProcess: false,
  },
};

/** Default cache directory for downloaded models. */
function getDefaultCacheDir(): string {
  return path.join(os.homedir(), ".liteparse", "models");
}

/**
 * Get the path to a layout model ONNX file, downloading if needed.
 *
 * @param modelNameOrPath - Either a registered model name or a direct file path to an ONNX file.
 * @param cacheDir - Override the default cache directory.
 * @returns Absolute path to the ONNX model file.
 */
export async function getModelPath(
  modelNameOrPath?: string,
  cacheDir?: string
): Promise<string> {
  const input = modelNameOrPath || DEFAULT_MODEL;

  // If it's an absolute path to an existing file, use it directly
  if (path.isAbsolute(input) && existsSync(input)) {
    return input;
  }

  // If it's a relative path to an existing file, resolve and use it
  if (existsSync(input)) {
    return path.resolve(input);
  }

  // Otherwise treat it as a model name from the registry
  const modelInfo = MODEL_REGISTRY[input];
  if (!modelInfo) {
    throw new Error(
      `Unknown layout model: "${input}". Available models: ${Object.keys(MODEL_REGISTRY).join(", ")}. ` +
        `Or provide a direct path to an ONNX file.`
    );
  }

  const dir = cacheDir || getDefaultCacheDir();
  const modelDir = path.join(dir, input);
  const modelPath = path.join(modelDir, modelInfo.filename);

  if (existsSync(modelPath)) {
    return modelPath;
  }

  // Download the model
  await fs.mkdir(modelDir, { recursive: true });
  await downloadModel(modelInfo.url, modelDir, modelInfo.filename);

  return modelPath;
}

/**
 * Get model metadata (number of classes, input size, labels).
 */
export function getModelMeta(modelNameOrPath?: string) {
  const input = modelNameOrPath || DEFAULT_MODEL;

  // Check registry by name
  const info = MODEL_REGISTRY[input];
  if (info) {
    return {
      numClasses: info.numClasses,
      inputSize: info.inputSize,
      labels: info.labels,
      hasPostProcess: info.hasPostProcess,
    };
  }

  // For file paths, try to match by path substring against registry entries
  for (const [key, regInfo] of Object.entries(MODEL_REGISTRY)) {
    if (input.includes(key)) {
      return {
        numClasses: regInfo.numClasses,
        inputSize: regInfo.inputSize,
        labels: regInfo.labels,
        hasPostProcess: regInfo.hasPostProcess,
      };
    }
  }

  // Fallback for completely unknown custom models — use 17cls defaults
  return {
    numClasses: 17,
    inputSize: [480, 480] as [number, number],
    labels: [
      "paragraph_title", "image", "text", "number", "abstract",
      "content", "figure_title", "formula", "table", "table_title",
      "reference", "doc_title", "footnote", "header", "algorithm",
      "footer", "seal",
    ],
    hasPostProcess: true,
  };
}

/**
 * Download a model file. Supports direct ONNX downloads and .tar archives.
 */
async function downloadModel(
  url: string,
  destDir: string,
  filename: string
): Promise<void> {
  console.error(`Downloading layout model from ${url}...`);

  const response = await axios.get(url, {
    responseType: "arraybuffer",
    onDownloadProgress: (progressEvent) => {
      if (progressEvent.total) {
        const pct = Math.round((progressEvent.loaded / progressEvent.total) * 100);
        process.stderr.write(`\r  Downloading: ${pct}%`);
      }
    },
  });

  console.error(""); // newline after progress

  const data = Buffer.from(response.data);

  if (url.endsWith(".tar")) {
    // Extract tar archive — look for .onnx or .pdmodel files
    await extractTar(data, destDir, filename);
  } else {
    // Direct file download
    await fs.writeFile(path.join(destDir, filename), data);
  }

  console.error(`  Model cached at ${destDir}`);
}

/**
 * Simple tar extraction — finds the target file pattern in a tar archive.
 * Tar format: 512-byte header blocks followed by file data blocks.
 */
async function extractTar(
  tarData: Buffer,
  destDir: string,
  targetFilename: string
): Promise<void> {
  let offset = 0;
  const files: { name: string; data: Buffer }[] = [];

  while (offset < tarData.length - 512) {
    // Read header
    const header = tarData.subarray(offset, offset + 512);

    // Check for end-of-archive (two zero blocks)
    if (header.every((b) => b === 0)) break;

    // Extract filename (bytes 0-99)
    const nameEnd = header.indexOf(0);
    const name = header.subarray(0, Math.min(nameEnd, 100)).toString("ascii").trim();

    // Extract file size (bytes 124-135, octal)
    const sizeStr = header.subarray(124, 136).toString("ascii").trim();
    const size = parseInt(sizeStr, 8) || 0;

    offset += 512; // Move past header

    if (size > 0 && name.length > 0) {
      const fileData = tarData.subarray(offset, offset + size);
      files.push({ name, data: fileData });
    }

    // Move past file data (padded to 512-byte boundary)
    offset += Math.ceil(size / 512) * 512;
  }

  // Look for the pdmodel file (we'll need to convert) or onnx file
  const onnxFile = files.find((f) => f.name.endsWith(".onnx"));
  const pdmodelFile = files.find((f) => f.name.endsWith(".pdmodel"));
  const pdiparamsFile = files.find((f) => f.name.endsWith(".pdiparams"));

  if (onnxFile) {
    await fs.writeFile(path.join(destDir, targetFilename), onnxFile.data);
  } else if (pdmodelFile && pdiparamsFile) {
    // Save paddle format files — user will need to convert to ONNX
    await fs.writeFile(path.join(destDir, "model.pdmodel"), pdmodelFile.data);
    await fs.writeFile(path.join(destDir, "model.pdiparams"), pdiparamsFile.data);
    throw new Error(
      `Downloaded model is in PaddlePaddle format. Convert to ONNX first:\n` +
        `  pip install paddle2onnx\n` +
        `  paddle2onnx --model_dir ${destDir} --model_filename model.pdmodel ` +
        `--params_filename model.pdiparams --opset_version 11 --save_file ${path.join(destDir, targetFilename)}`
    );
  } else {
    // Save all files for debugging
    for (const f of files) {
      const safeName = path.basename(f.name);
      if (safeName) {
        await fs.writeFile(path.join(destDir, safeName), f.data);
      }
    }
    throw new Error(
      `No ONNX model found in archive. Extracted files: ${files.map((f) => f.name).join(", ")}. ` +
        `You may need to convert the model to ONNX format manually.`
    );
  }
}
