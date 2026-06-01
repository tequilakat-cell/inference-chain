import type { ThoughtDocument, RollupDocument } from "./types.js";

export function cosineSim(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

export function textMatchThoughts(
  docs: ThoughtDocument[],
  q: string,
  modelId: string,
  limit: number,
): Array<ThoughtDocument & { score: number }> {
  const ql = q.toLowerCase();
  return docs
    .filter(d => {
      if (modelId && d.model_id !== modelId) return false;
      return (
        d.question_text?.toLowerCase().includes(ql) ||
        d.answer_text?.toLowerCase().includes(ql) ||
        d.thinking_text?.toLowerCase().includes(ql)
      );
    })
    .slice(0, limit)
    .map(d => ({ ...d, score: 1.0 }));
}

export function semanticSearchThoughts(
  docs: ThoughtDocument[],
  embedding: number[],
  modelId: string,
  limit: number,
  minScore = 0.0,
): Array<ThoughtDocument & { score: number }> {
  return docs
    .filter(d => d.embedding && d.embedding.length > 0 && (!modelId || d.model_id === modelId))
    .map(d => ({ ...d, score: cosineSim(embedding, d.embedding!) }))
    .filter(d => d.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

export function semanticSearchRollups(
  docs: RollupDocument[],
  embedding: number[],
  modelId: string,
  limit: number,
  minScore = 0.0,
): Array<RollupDocument & { score: number }> {
  return docs
    .filter(d => d.embedding && d.embedding.length > 0 && (!modelId || d.model_id === modelId))
    .map(d => ({ ...d, score: cosineSim(embedding, d.embedding!) }))
    .filter(d => d.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
