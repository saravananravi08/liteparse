/**
 * OpenAI-compatible vision engine.
 *
 * Works with any API that implements the OpenAI chat completions format:
 * - OpenAI (GPT-4o, GPT-4o-mini)
 * - Anthropic (via proxy or compatible endpoint)
 * - Ollama (local models with vision)
 * - Any OpenAI-compatible provider
 */

import axios from "axios";
import { LayoutLabel } from "../layout/interface.js";
import { VisionEngine, DEFAULT_VISION_PROMPTS } from "./interface.js";

export interface OpenAIVisionOptions {
  /** API key for authentication. */
  apiKey: string;
  /** API base URL. Defaults to OpenAI. */
  apiUrl?: string;
  /** Model name. Defaults to "gpt-4o". */
  model?: string;
  /** Max tokens for the response. */
  maxTokens?: number;
}

export class OpenAIVisionEngine implements VisionEngine {
  name = "OpenAI Vision";

  private apiKey: string;
  private apiUrl: string;
  private model: string;
  private maxTokens: number;

  constructor(options: OpenAIVisionOptions) {
    this.apiKey = options.apiKey;
    this.apiUrl = (options.apiUrl || "https://api.openai.com/v1").replace(/\/$/, "");
    this.model = options.model || "gpt-4o";
    this.maxTokens = options.maxTokens || 1024;
  }

  async describe(
    image: Buffer,
    regionType: LayoutLabel,
    customPrompt?: string
  ): Promise<string> {
    const prompt = customPrompt || DEFAULT_VISION_PROMPTS[regionType] || DEFAULT_VISION_PROMPTS.image!;

    const base64Image = image.toString("base64");

    const response = await axios.post(
      `${this.apiUrl}/chat/completions`,
      {
        model: this.model,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/png;base64,${base64Image}`,
                  detail: "high",
                },
              },
            ],
          },
        ],
        max_tokens: this.maxTokens,
      },
      {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        timeout: 60000,
      }
    );

    return response.data.choices[0]?.message?.content?.trim() || "";
  }
}
