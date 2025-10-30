/**
 * Type definitions for AI Memory SDK
 */
interface ClientConfig {
    apiKey?: string;
    baseUrl?: string;
    timeout?: number;
}
interface AuthResponse {
    access_token: string;
    token_type: string;
    user_id: string;
}
interface User {
    id: string;
    email: string;
    full_name: string;
    is_active: boolean;
    is_verified: boolean;
    tier: 'free' | 'starter' | 'pro' | 'enterprise';
}
interface Collection {
    id: string;
    name: string;
    description?: string;
    is_active: boolean;
    memory_count: number;
    metadata: Record<string, any>;
    created_at: string;
    updated_at: string;
}
interface Memory {
    id: string;
    collection_id: string;
    content: string;
    importance: number;
    source_type: string;
    source_reference?: string;
    access_count: number;
    last_accessed_at?: string;
    created_at: string;
    updated_at: string;
}
interface MemoryWithMetadata extends Memory {
    metadata: Record<string, any>;
}
interface SearchResult {
    memory_id: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
    created_at: string;
}
interface SearchResponse {
    query: string;
    results: SearchResult[];
    total: number;
    search_type: string;
    processing_time_ms: number;
}
interface ReasoningSource {
    memory_id: string;
    content: string;
    score: number;
}
interface ReasoningResponse {
    answer: string;
    sources: ReasoningSource[];
    metadata: Record<string, any>;
    reasoning_context?: Record<string, any>;
}
interface CreateCollectionParams {
    name: string;
    description?: string;
    metadata?: Record<string, any>;
}
interface UpdateCollectionParams {
    name?: string;
    description?: string;
    metadata?: Record<string, any>;
}
interface CreateMemoryParams {
    collection_id: string;
    content: string;
    importance?: number;
    source_type?: string;
    source_reference?: string;
    metadata?: Record<string, any>;
}
interface UpdateMemoryParams {
    content?: string;
    importance?: number;
    metadata?: Record<string, any>;
}
interface SearchParams {
    query: string;
    collection_id?: string;
    limit?: number;
    search_type?: 'hybrid' | 'vector' | 'bm25' | 'graph';
    filters?: Record<string, any>;
}
interface ReasonParams {
    query: string;
    collection_id?: string;
    provider?: 'gemini' | 'openai' | 'anthropic';
    include_steps?: boolean;
}
interface ListParams {
    skip?: number;
    limit?: number;
}
interface APIKeyResponse {
    id: string;
    name: string;
    key: string;
    prefix: string;
    created_at: string;
}

/**
 * API Resource classes
 */

declare class BaseResource {
    protected client: MemoryClient;
    constructor(client: MemoryClient);
}
declare class AuthResource extends BaseResource {
    /**
     * Register a new user
     */
    register(email: string, password: string, fullName: string): Promise<AuthResponse>;
    /**
     * Login with email and password
     */
    login(email: string, password: string): Promise<AuthResponse>;
    /**
     * Create a new API key
     */
    createApiKey(name: string): Promise<APIKeyResponse>;
    /**
     * Get current user information
     */
    getMe(): Promise<User>;
}
declare class CollectionsResource extends BaseResource {
    /**
     * Create a new collection
     */
    create(params: CreateCollectionParams): Promise<Collection>;
    /**
     * List all collections
     */
    list(params?: ListParams): Promise<Collection[]>;
    /**
     * Get a specific collection
     */
    get(collectionId: string): Promise<Collection>;
    /**
     * Update a collection
     */
    update(collectionId: string, params: UpdateCollectionParams): Promise<Collection>;
    /**
     * Delete a collection
     */
    delete(collectionId: string): Promise<void>;
}
declare class MemoriesResource extends BaseResource {
    /**
     * Create a new memory
     */
    create(params: CreateMemoryParams): Promise<Memory>;
    /**
     * List memories
     */
    list(params?: ListParams & {
        collection_id?: string;
    }): Promise<Memory[]>;
    /**
     * Get a specific memory
     */
    get(memoryId: string): Promise<MemoryWithMetadata>;
    /**
     * Update a memory
     */
    update(memoryId: string, params: UpdateMemoryParams): Promise<Memory>;
    /**
     * Delete a memory
     */
    delete(memoryId: string): Promise<void>;
}
declare class SearchResource extends BaseResource {
    /**
     * Search memories
     */
    search(params: SearchParams): Promise<SearchResult[]>;
    /**
     * Find similar memories
     */
    similar(memoryId: string, limit?: number): Promise<SearchResult[]>;
    /**
     * Perform reasoning over memories
     */
    reason(params: ReasonParams): Promise<ReasoningResponse>;
}
declare class RLResource extends BaseResource {
    /**
     * Train the Memory Manager agent using RL
     */
    trainMemoryManager(params: {
        collection_id?: string;
        num_episodes?: number;
        [key: string]: any;
    }): Promise<any>;
    /**
     * Train the Answer Agent using RL
     */
    trainAnswerAgent(params: {
        collection_id?: string;
        num_episodes?: number;
        [key: string]: any;
    }): Promise<any>;
    /**
     * Get RL training metrics
     */
    getMetrics(): Promise<any>;
    /**
     * Evaluate a trained RL agent
     */
    evaluate(agentType: string, collectionId?: string): Promise<any>;
}
declare class ProceduralResource extends BaseResource {
    /**
     * Create a new procedure
     */
    create(params: {
        name: string;
        description: string;
        trigger_condition: string;
        action_sequence: string[];
        collection_id?: string;
        category?: string;
        metadata?: Record<string, any>;
    }): Promise<any>;
    /**
     * List procedures
     */
    list(params?: {
        collection_id?: string;
        category?: string;
        skip?: number;
        limit?: number;
    }): Promise<any[]>;
    /**
     * Get a specific procedure
     */
    get(procedureId: string): Promise<any>;
    /**
     * Execute a procedure
     */
    execute(procedureId: string, context?: Record<string, any>): Promise<any>;
    /**
     * Update a procedure
     */
    update(procedureId: string, params: {
        name?: string;
        description?: string;
        trigger_condition?: string;
        action_sequence?: string[];
        metadata?: Record<string, any>;
    }): Promise<any>;
    /**
     * Delete a procedure
     */
    delete(procedureId: string): Promise<void>;
}
declare class TemporalResource extends BaseResource {
    /**
     * Add a temporal fact to the knowledge graph
     */
    addFact(params: {
        subject: string;
        predicate: string;
        object: string;
        valid_from?: string;
        valid_until?: string;
        confidence?: number;
        source_memory_id?: string;
        metadata?: Record<string, any>;
    }): Promise<any>;
    /**
     * Query temporal facts
     */
    queryFacts(params?: {
        subject?: string;
        predicate?: string;
        object?: string;
        at_time?: string;
    }): Promise<any[]>;
    /**
     * Query knowledge state at a specific point in time
     */
    pointInTime(timestamp: string, entity?: string): Promise<any>;
}
declare class WorkingMemoryResource extends BaseResource {
    /**
     * Add item to working memory buffer
     */
    add(params: {
        role: string;
        content: string;
        metadata?: Record<string, any>;
    }): Promise<any>;
    /**
     * Get current working memory context
     */
    getContext(): Promise<any>;
    /**
     * Compress working memory buffer
     */
    compress(): Promise<any>;
    /**
     * Clear working memory buffer
     */
    clear(): Promise<any>;
}
declare class ConsolidationResource extends BaseResource {
    /**
     * Trigger memory consolidation
     */
    consolidate(collectionId: string, threshold?: number): Promise<any>;
    /**
     * Get consolidation statistics
     */
    getStats(collectionId: string): Promise<any>;
    /**
     * Archive old memories
     */
    archive(collectionId: string, beforeDate?: string): Promise<any>;
}
declare class MemoryToolsResource extends BaseResource {
    /**
     * Replace memory content
     */
    replace(params: {
        memory_id: string;
        new_content: string;
        reason?: string;
    }): Promise<any>;
    /**
     * Insert new memory at position
     */
    insert(params: {
        collection_id: string;
        content: string;
        position: number;
        reason?: string;
    }): Promise<any>;
    /**
     * Re-evaluate memory in light of new information
     */
    rethink(memoryId: string, query: string): Promise<any>;
}
declare class WorldModelResource extends BaseResource {
    /**
     * Simulate retrieval without actually retrieving
     */
    imagineRetrieval(query: string, collectionId?: string): Promise<any>;
    /**
     * Plan memory operations to achieve goal
     */
    plan(goal: string, collectionId?: string): Promise<any>;
}

/**
 * Main Memory AI client
 */

declare class MemoryClient {
    private apiKey?;
    private baseUrl;
    private timeout;
    auth: AuthResource;
    collections: CollectionsResource;
    memories: MemoriesResource;
    private searchResource;
    rl: RLResource;
    procedural: ProceduralResource;
    temporal: TemporalResource;
    workingMemory: WorkingMemoryResource;
    consolidation: ConsolidationResource;
    memoryTools: MemoryToolsResource;
    worldModel: WorldModelResource;
    constructor(config?: ClientConfig);
    private getHeaders;
    private handleError;
    request<T = any>(method: string, path: string, options?: RequestInit): Promise<T>;
    get<T = any>(path: string, params?: Record<string, any>): Promise<T>;
    post<T = any>(path: string, body?: any): Promise<T>;
    patch<T = any>(path: string, body?: any): Promise<T>;
    delete<T = any>(path: string): Promise<T>;
    search(params: SearchParams): Promise<SearchResult[]>;
    reason(params: ReasonParams): Promise<ReasoningResponse>;
}

/**
 * Error classes for AI Memory SDK
 */
declare class MemoryAIError extends Error {
    statusCode?: number;
    constructor(message: string, statusCode?: number);
}
declare class AuthenticationError extends MemoryAIError {
    constructor(message?: string);
}
declare class ValidationError extends MemoryAIError {
    errors?: any[];
    constructor(message: string, errors?: any[]);
}
declare class NotFoundError extends MemoryAIError {
    constructor(message?: string);
}
declare class RateLimitError extends MemoryAIError {
    constructor(message?: string);
}
declare class ServerError extends MemoryAIError {
    constructor(message?: string);
}

export { type APIKeyResponse, AuthResource, type AuthResponse, AuthenticationError, type ClientConfig, type Collection, CollectionsResource, ConsolidationResource, type CreateCollectionParams, type CreateMemoryParams, type ListParams, MemoriesResource, type Memory, MemoryAIError, MemoryClient, MemoryToolsResource, type MemoryWithMetadata, NotFoundError, ProceduralResource, RLResource, RateLimitError, type ReasonParams, type ReasoningResponse, type ReasoningSource, type SearchParams, SearchResource, type SearchResponse, type SearchResult, ServerError, TemporalResource, type UpdateCollectionParams, type UpdateMemoryParams, type User, ValidationError, WorkingMemoryResource, WorldModelResource };
