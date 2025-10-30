/**
 * AI Memory TypeScript SDK
 *
 * Official TypeScript/JavaScript client library for the AI Memory API.
 *
 * @example
 * ```typescript
 * import { MemoryClient } from '@memory-ai/sdk';
 *
 * const client = new MemoryClient({ apiKey: 'mem_sk_...' });
 *
 * // Create a memory
 * const memory = await client.memories.create({
 *   collectionId: 'col_123',
 *   content: 'Important information'
 * });
 *
 * // Search memories
 * const results = await client.search({
 *   query: 'important',
 *   limit: 5
 * });
 * ```
 */

export { MemoryClient } from './client';
export * from './types';
export * from './errors';
export * from './resources';
