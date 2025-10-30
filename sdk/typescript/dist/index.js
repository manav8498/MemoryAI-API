"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  AuthResource: () => AuthResource,
  AuthenticationError: () => AuthenticationError,
  CollectionsResource: () => CollectionsResource,
  ConsolidationResource: () => ConsolidationResource,
  MemoriesResource: () => MemoriesResource,
  MemoryAIError: () => MemoryAIError,
  MemoryClient: () => MemoryClient,
  MemoryToolsResource: () => MemoryToolsResource,
  NotFoundError: () => NotFoundError,
  ProceduralResource: () => ProceduralResource,
  RLResource: () => RLResource,
  RateLimitError: () => RateLimitError,
  SearchResource: () => SearchResource,
  ServerError: () => ServerError,
  TemporalResource: () => TemporalResource,
  ValidationError: () => ValidationError,
  WorkingMemoryResource: () => WorkingMemoryResource,
  WorldModelResource: () => WorldModelResource
});
module.exports = __toCommonJS(index_exports);

// src/resources.ts
var BaseResource = class {
  constructor(client) {
    this.client = client;
  }
};
var AuthResource = class extends BaseResource {
  /**
   * Register a new user
   */
  async register(email, password, fullName) {
    return this.client.post("/v1/auth/register", {
      email,
      password,
      full_name: fullName
    });
  }
  /**
   * Login with email and password
   */
  async login(email, password) {
    return this.client.post("/v1/auth/login", {
      email,
      password
    });
  }
  /**
   * Create a new API key
   */
  async createApiKey(name) {
    return this.client.post("/v1/auth/api-keys", { name });
  }
  /**
   * Get current user information
   */
  async getMe() {
    return this.client.get("/v1/auth/me");
  }
};
var CollectionsResource = class extends BaseResource {
  /**
   * Create a new collection
   */
  async create(params) {
    return this.client.post("/v1/collections", params);
  }
  /**
   * List all collections
   */
  async list(params = {}) {
    return this.client.get("/v1/collections", params);
  }
  /**
   * Get a specific collection
   */
  async get(collectionId) {
    return this.client.get(`/v1/collections/${collectionId}`);
  }
  /**
   * Update a collection
   */
  async update(collectionId, params) {
    return this.client.patch(
      `/v1/collections/${collectionId}`,
      params
    );
  }
  /**
   * Delete a collection
   */
  async delete(collectionId) {
    await this.client.delete(`/v1/collections/${collectionId}`);
  }
};
var MemoriesResource = class extends BaseResource {
  /**
   * Create a new memory
   */
  async create(params) {
    return this.client.post("/v1/memories", params);
  }
  /**
   * List memories
   */
  async list(params = {}) {
    return this.client.get("/v1/memories", params);
  }
  /**
   * Get a specific memory
   */
  async get(memoryId) {
    return this.client.get(`/v1/memories/${memoryId}`);
  }
  /**
   * Update a memory
   */
  async update(memoryId, params) {
    return this.client.patch(`/v1/memories/${memoryId}`, params);
  }
  /**
   * Delete a memory
   */
  async delete(memoryId) {
    await this.client.delete(`/v1/memories/${memoryId}`);
  }
};
var SearchResource = class extends BaseResource {
  /**
   * Search memories
   */
  async search(params) {
    const response = await this.client.post("/v1/search", {
      query: params.query,
      collection_id: params.collection_id,
      limit: params.limit || 10,
      search_type: params.search_type || "hybrid",
      filters: params.filters || {}
    });
    return response.results;
  }
  /**
   * Find similar memories
   */
  async similar(memoryId, limit = 10) {
    const response = await this.client.get(
      `/v1/search/similar/${memoryId}`,
      { limit }
    );
    return response.results;
  }
  /**
   * Perform reasoning over memories
   */
  async reason(params) {
    return this.client.post("/v1/search/reason", {
      query: params.query,
      collection_id: params.collection_id,
      provider: params.provider,
      include_steps: params.include_steps || false
    });
  }
};
var RLResource = class extends BaseResource {
  /**
   * Train the Memory Manager agent using RL
   */
  async trainMemoryManager(params) {
    return this.client.post("/rl/train/memory-manager", {
      collection_id: params.collection_id,
      num_episodes: params.num_episodes || 100,
      ...params
    });
  }
  /**
   * Train the Answer Agent using RL
   */
  async trainAnswerAgent(params) {
    return this.client.post("/rl/train/answer-agent", {
      collection_id: params.collection_id,
      num_episodes: params.num_episodes || 100,
      ...params
    });
  }
  /**
   * Get RL training metrics
   */
  async getMetrics() {
    return this.client.get("/rl/metrics");
  }
  /**
   * Evaluate a trained RL agent
   */
  async evaluate(agentType, collectionId) {
    return this.client.post("/rl/evaluate", {
      agent_type: agentType,
      collection_id: collectionId
    });
  }
};
var ProceduralResource = class extends BaseResource {
  /**
   * Create a new procedure
   */
  async create(params) {
    return this.client.post("/procedural", {
      name: params.name,
      description: params.description,
      trigger_condition: params.trigger_condition,
      action_sequence: params.action_sequence,
      collection_id: params.collection_id,
      category: params.category,
      metadata: params.metadata || {}
    });
  }
  /**
   * List procedures
   */
  async list(params) {
    return this.client.get("/procedural", {
      collection_id: params?.collection_id,
      category: params?.category,
      skip: params?.skip || 0,
      limit: params?.limit || 100
    });
  }
  /**
   * Get a specific procedure
   */
  async get(procedureId) {
    return this.client.get(`/procedural/${procedureId}`);
  }
  /**
   * Execute a procedure
   */
  async execute(procedureId, context) {
    return this.client.post(`/procedural/${procedureId}/execute`, {
      context: context || {}
    });
  }
  /**
   * Update a procedure
   */
  async update(procedureId, params) {
    return this.client.patch(`/procedural/${procedureId}`, params);
  }
  /**
   * Delete a procedure
   */
  async delete(procedureId) {
    await this.client.delete(`/procedural/${procedureId}`);
  }
};
var TemporalResource = class extends BaseResource {
  /**
   * Add a temporal fact to the knowledge graph
   */
  async addFact(params) {
    return this.client.post("/temporal/facts", {
      subject: params.subject,
      predicate: params.predicate,
      object: params.object,
      valid_from: params.valid_from,
      valid_until: params.valid_until,
      confidence: params.confidence || 1,
      source_memory_id: params.source_memory_id,
      metadata: params.metadata || {}
    });
  }
  /**
   * Query temporal facts
   */
  async queryFacts(params) {
    return this.client.get("/temporal/facts", {
      subject: params?.subject,
      predicate: params?.predicate,
      object: params?.object,
      at_time: params?.at_time
    });
  }
  /**
   * Query knowledge state at a specific point in time
   */
  async pointInTime(timestamp, entity) {
    return this.client.post("/temporal/point-in-time", {
      timestamp,
      entity
    });
  }
};
var WorkingMemoryResource = class extends BaseResource {
  /**
   * Add item to working memory buffer
   */
  async add(params) {
    return this.client.post("/working-memory", {
      role: params.role,
      content: params.content,
      metadata: params.metadata || {}
    });
  }
  /**
   * Get current working memory context
   */
  async getContext() {
    return this.client.get("/working-memory/context");
  }
  /**
   * Compress working memory buffer
   */
  async compress() {
    return this.client.post("/working-memory/compress");
  }
  /**
   * Clear working memory buffer
   */
  async clear() {
    return this.client.delete("/working-memory");
  }
};
var ConsolidationResource = class extends BaseResource {
  /**
   * Trigger memory consolidation
   */
  async consolidate(collectionId, threshold = 100) {
    return this.client.post("/consolidation/consolidate", {
      collection_id: collectionId,
      threshold
    });
  }
  /**
   * Get consolidation statistics
   */
  async getStats(collectionId) {
    return this.client.get("/consolidation/stats", {
      collection_id: collectionId
    });
  }
  /**
   * Archive old memories
   */
  async archive(collectionId, beforeDate) {
    return this.client.post("/consolidation/archive", {
      collection_id: collectionId,
      before_date: beforeDate
    });
  }
};
var MemoryToolsResource = class extends BaseResource {
  /**
   * Replace memory content
   */
  async replace(params) {
    return this.client.post("/memory-tools/replace", {
      memory_id: params.memory_id,
      new_content: params.new_content,
      reason: params.reason
    });
  }
  /**
   * Insert new memory at position
   */
  async insert(params) {
    return this.client.post("/memory-tools/insert", {
      collection_id: params.collection_id,
      content: params.content,
      position: params.position,
      reason: params.reason
    });
  }
  /**
   * Re-evaluate memory in light of new information
   */
  async rethink(memoryId, query) {
    return this.client.post("/memory-tools/rethink", {
      memory_id: memoryId,
      query
    });
  }
};
var WorldModelResource = class extends BaseResource {
  /**
   * Simulate retrieval without actually retrieving
   */
  async imagineRetrieval(query, collectionId) {
    return this.client.post("/world-model/imagine-retrieval", {
      query,
      collection_id: collectionId
    });
  }
  /**
   * Plan memory operations to achieve goal
   */
  async plan(goal, collectionId) {
    return this.client.post("/world-model/plan", {
      goal,
      collection_id: collectionId
    });
  }
};

// src/errors.ts
var MemoryAIError = class _MemoryAIError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = "MemoryAIError";
    this.statusCode = statusCode;
    Object.setPrototypeOf(this, _MemoryAIError.prototype);
  }
};
var AuthenticationError = class _AuthenticationError extends MemoryAIError {
  constructor(message = "Authentication failed") {
    super(message, 401);
    this.name = "AuthenticationError";
    Object.setPrototypeOf(this, _AuthenticationError.prototype);
  }
};
var ValidationError = class _ValidationError extends MemoryAIError {
  constructor(message, errors) {
    super(message, 422);
    this.name = "ValidationError";
    this.errors = errors;
    Object.setPrototypeOf(this, _ValidationError.prototype);
  }
};
var NotFoundError = class _NotFoundError extends MemoryAIError {
  constructor(message = "Resource not found") {
    super(message, 404);
    this.name = "NotFoundError";
    Object.setPrototypeOf(this, _NotFoundError.prototype);
  }
};
var RateLimitError = class _RateLimitError extends MemoryAIError {
  constructor(message = "Rate limit exceeded") {
    super(message, 429);
    this.name = "RateLimitError";
    Object.setPrototypeOf(this, _RateLimitError.prototype);
  }
};
var ServerError = class _ServerError extends MemoryAIError {
  constructor(message = "Internal server error") {
    super(message, 500);
    this.name = "ServerError";
    Object.setPrototypeOf(this, _ServerError.prototype);
  }
};

// src/client.ts
var MemoryClient = class {
  constructor(config = {}) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || "http://localhost:8000";
    this.timeout = config.timeout || 3e4;
    this.baseUrl = this.baseUrl.replace(/\/$/, "");
    this.auth = new AuthResource(this);
    this.collections = new CollectionsResource(this);
    this.memories = new MemoriesResource(this);
    this.searchResource = new SearchResource(this);
    this.rl = new RLResource(this);
    this.procedural = new ProceduralResource(this);
    this.temporal = new TemporalResource(this);
    this.workingMemory = new WorkingMemoryResource(this);
    this.consolidation = new ConsolidationResource(this);
    this.memoryTools = new MemoryToolsResource(this);
    this.worldModel = new WorldModelResource(this);
  }
  getHeaders() {
    const headers = {
      "Content-Type": "application/json",
      "User-Agent": "memory-ai-typescript/1.0.0"
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }
  handleError(response, data) {
    const statusCode = response.status;
    const message = data?.error?.message || data?.detail || response.statusText;
    if (statusCode === 401) {
      throw new AuthenticationError(message);
    } else if (statusCode === 404) {
      throw new NotFoundError(message);
    } else if (statusCode === 422) {
      const errors = data?.error?.details || [];
      throw new ValidationError(message, errors);
    } else if (statusCode === 429) {
      throw new RateLimitError(message);
    } else if (statusCode >= 500) {
      throw new ServerError(message);
    } else {
      throw new MemoryAIError(message, statusCode);
    }
  }
  async request(method, path, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      method,
      headers: {
        ...this.getHeaders(),
        ...options.headers || {}
      },
      ...options,
      signal: AbortSignal.timeout(this.timeout)
    });
    let data;
    const contentType = response.headers.get("content-type");
    if (contentType?.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }
    if (!response.ok) {
      this.handleError(response, data);
    }
    return data;
  }
  async get(path, params) {
    let url = path;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== void 0 && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      url += `?${searchParams.toString()}`;
    }
    return this.request("GET", url);
  }
  async post(path, body) {
    return this.request("POST", path, {
      body: JSON.stringify(body)
    });
  }
  async patch(path, body) {
    return this.request("PATCH", path, {
      body: JSON.stringify(body)
    });
  }
  async delete(path) {
    return this.request("DELETE", path);
  }
  // Convenience methods
  async search(params) {
    return this.searchResource.search(params);
  }
  async reason(params) {
    return this.searchResource.reason(params);
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  AuthResource,
  AuthenticationError,
  CollectionsResource,
  ConsolidationResource,
  MemoriesResource,
  MemoryAIError,
  MemoryClient,
  MemoryToolsResource,
  NotFoundError,
  ProceduralResource,
  RLResource,
  RateLimitError,
  SearchResource,
  ServerError,
  TemporalResource,
  ValidationError,
  WorkingMemoryResource,
  WorldModelResource
});
